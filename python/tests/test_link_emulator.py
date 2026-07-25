from __future__ import annotations

import socket
import threading
import unittest

from distributed_runtime.protocol import (
    HEADER,
    HEADER_BYTES,
    FrameType,
    LinkEmulator,
    LinkEmulatorError,
    recv_frame,
    send_frame,
)


class _ManualClock:
    def __init__(self, initial: float = 0.0) -> None:
        self._value = initial
        self._reads = 0
        self._condition = threading.Condition()

    def __call__(self) -> float:
        with self._condition:
            self._reads += 1
            self._condition.notify_all()
            return self._value

    @property
    def value(self) -> float:
        with self._condition:
            return self._value

    @property
    def reads(self) -> int:
        with self._condition:
            return self._reads

    def advance(self, seconds: float) -> None:
        with self._condition:
            self._value += seconds
            self._condition.notify_all()

    def wait_for_read_after(self, previous: int, timeout: float = 1.0) -> None:
        with self._condition:
            if not self._condition.wait_for(
                lambda: self._reads > previous,
                timeout=timeout,
            ):
                raise AssertionError("emulated-link worker did not read the clock")


class _RecordingSocket:
    def __init__(self, clock: _ManualClock, *, fail: bool = False) -> None:
        self.clock = clock
        self.fail = fail
        self.chunks: list[bytes] = []
        self.send_times: list[float] = []
        self.send_threads: list[str] = []
        self.shutdown_calls = 0
        self._condition = threading.Condition()

    def sendall(self, data: bytes | bytearray | memoryview) -> None:
        with self._condition:
            self.send_times.append(self.clock.value)
            self.send_threads.append(threading.current_thread().name)
            if self.fail:
                self._condition.notify_all()
                raise OSError("recording socket exploded")
            self.chunks.append(bytes(data))
            self._condition.notify_all()

    def shutdown(self, _: int) -> None:
        with self._condition:
            self.shutdown_calls += 1
            self._condition.notify_all()

    def wait_for_calls(self, count: int, timeout: float = 1.0) -> None:
        with self._condition:
            if not self._condition.wait_for(
                lambda: len(self.send_times) >= count,
                timeout=timeout,
            ):
                raise AssertionError(
                    f"expected {count} socket writes, got {len(self.send_times)}"
                )


def _advance_and_wake(
    clock: _ManualClock,
    emulator: LinkEmulator,
    seconds: float,
) -> None:
    previous_reads = clock.reads
    clock.advance(seconds)
    emulator.notify_clock_advanced()
    clock.wait_for_read_after(previous_reads)


class LinkEmulatorTests(unittest.TestCase):
    def test_disabled_emulator_and_none_keep_inline_send_path(self) -> None:
        clock = _ManualClock()
        direct = _RecordingSocket(clock)
        inactive = LinkEmulator(clock=clock)
        expected_thread = threading.current_thread().name

        send_frame(direct, FrameType.BEGIN, 1)
        send_frame(direct, FrameType.END, 1, emulator=inactive)

        self.assertEqual(len(direct.chunks), 2)
        self.assertEqual(direct.send_threads, [expected_thread, expected_thread])
        self.assertEqual(inactive.pending_frames, 0)
        inactive.close()

    def test_one_frame_matches_propagation_plus_serialization_model(self) -> None:
        clock = _ManualClock(initial=100.0)
        # A 32-byte frame takes exactly 250 ms at this bandwidth.
        bandwidth_mbps = (HEADER_BYTES * 8) / (0.25 * 1_000_000)
        emulator = LinkEmulator(
            one_way_delay_ms=500,
            bandwidth_mbps=bandwidth_mbps,
            clock=clock,
        )
        recorded = _RecordingSocket(clock)

        send_frame(recorded, FrameType.BEGIN, 7, emulator=emulator)
        _advance_and_wake(clock, emulator, 0.749)
        self.assertEqual(recorded.send_times, [])
        _advance_and_wake(clock, emulator, 0.001)
        recorded.wait_for_calls(1)
        emulator.flush(timeout_seconds=1)

        self.assertEqual(recorded.send_times, [100.75])
        self.assertAlmostEqual(
            emulator.single_frame_delay_seconds(HEADER_BYTES),
            0.75,
        )
        emulator.close(timeout_seconds=1)

    def test_two_frames_overlap_propagation_and_preserve_fifo_order(self) -> None:
        clock = _ManualClock()
        emulator = LinkEmulator(one_way_delay_ms=1_000, clock=clock)
        recorded = _RecordingSocket(clock)

        send_frame(recorded, FrameType.BEGIN, 11, emulator=emulator)
        send_frame(recorded, FrameType.BEGIN, 22, emulator=emulator)
        _advance_and_wake(clock, emulator, 1.0)
        recorded.wait_for_calls(2)
        emulator.flush(timeout_seconds=1)

        request_ids = [HEADER.unpack(chunk)[4] for chunk in recorded.chunks]
        self.assertEqual(request_ids, [11, 22])
        self.assertEqual(recorded.send_times, [1.0, 1.0])
        self.assertTrue(
            all(
                name.startswith("mycellios-link-sender-")
                for name in recorded.send_threads
            )
        )
        emulator.close(timeout_seconds=1)

    def test_control_frame_without_explicit_emulator_cannot_overtake_data(self) -> None:
        clock = _ManualClock()
        emulator = LinkEmulator(one_way_delay_ms=1_000, clock=clock)
        recorded = _RecordingSocket(clock)

        send_frame(
            recorded,
            FrameType.ACTIVATION,
            31,
            payload=b"\0\0\0\0",
            token_count=1,
            hidden_size=1,
            emulator=emulator,
        )
        # Real engine/stage call sites historically omitted the emulator for
        # lifecycle frames. Once the stream is owned this frame must enter the
        # same FIFO instead of calling sendall inline.
        send_frame(recorded, FrameType.END, 31)
        self.assertEqual(recorded.send_times, [])

        _advance_and_wake(clock, emulator, 1.0)
        recorded.wait_for_calls(2)
        emulator.close(timeout_seconds=1)
        self.assertEqual(
            [
                HEADER.unpack(chunk)[2]
                for chunk in recorded.chunks
                if len(chunk) == HEADER_BYTES
            ],
            [FrameType.ACTIVATION, FrameType.END],
        )

        # Successful close releases stream ownership; a later unrelated
        # non-emulated lifecycle write returns to the exact inline path.
        send_frame(recorded, FrameType.SHUTDOWN, 0)
        self.assertEqual(recorded.send_threads[-1], threading.current_thread().name)

    def test_rejected_first_frame_does_not_poison_socket_ownership(self) -> None:
        clock = _ManualClock()
        emulator = LinkEmulator(
            one_way_delay_ms=1_000,
            max_queued_bytes=HEADER_BYTES,
            clock=clock,
        )
        recorded = _RecordingSocket(clock)

        with self.assertRaisesRegex(ValueError, "queue byte capacity"):
            send_frame(
                recorded,
                FrameType.ACTIVATION,
                31,
                payload=b"\0\0\0\0",
                token_count=1,
                hidden_size=1,
                emulator=emulator,
            )

        send_frame(recorded, FrameType.BEGIN, 32)
        self.assertEqual(recorded.send_threads, [threading.current_thread().name])
        self.assertEqual(HEADER.unpack(recorded.chunks[0])[2], FrameType.BEGIN)
        emulator.close()

    def test_bandwidth_serialization_remains_ordered_while_delay_overlaps(self) -> None:
        clock = _ManualClock()
        # One second of serialization for each payload-free frame.
        bandwidth_mbps = (HEADER_BYTES * 8) / 1_000_000
        emulator = LinkEmulator(
            one_way_delay_ms=2_000,
            bandwidth_mbps=bandwidth_mbps,
            clock=clock,
        )
        recorded = _RecordingSocket(clock)

        send_frame(recorded, FrameType.BEGIN, 1, emulator=emulator)
        send_frame(recorded, FrameType.BEGIN, 2, emulator=emulator)
        _advance_and_wake(clock, emulator, 3.0)
        recorded.wait_for_calls(1)
        self.assertEqual(recorded.send_times, [3.0])
        _advance_and_wake(clock, emulator, 1.0)
        recorded.wait_for_calls(2)
        emulator.flush(timeout_seconds=1)

        self.assertEqual(recorded.send_times, [3.0, 4.0])
        self.assertEqual(
            [HEADER.unpack(chunk)[4] for chunk in recorded.chunks],
            [1, 2],
        )
        emulator.close(timeout_seconds=1)

    def test_real_socket_round_trip_is_byte_exact_after_async_drain(self) -> None:
        clock = _ManualClock()
        emulator = LinkEmulator(one_way_delay_ms=1_000, clock=clock)
        sender, receiver = socket.socketpair()
        try:
            send_frame(sender, FrameType.BEGIN, 91, emulator=emulator)
            send_frame(
                sender,
                FrameType.ERROR,
                91,
                payload=b"byte-exact",
                emulator=emulator,
            )
            _advance_and_wake(clock, emulator, 1.0)
            emulator.flush(timeout_seconds=1)

            begin = recv_frame(receiver)
            error = recv_frame(receiver)
            self.assertEqual(
                (begin.frame_type, begin.request_id), (FrameType.BEGIN, 91)
            )
            self.assertEqual(
                (error.frame_type, error.request_id), (FrameType.ERROR, 91)
            )
            self.assertEqual(error.payload, b"byte-exact")
            emulator.close(timeout_seconds=1)
        finally:
            sender.close()
            receiver.close()

    def test_queue_is_bounded_and_close_rejects_future_frames(self) -> None:
        clock = _ManualClock()
        emulator = LinkEmulator(
            one_way_delay_ms=1_000,
            max_queued_frames=1,
            enqueue_timeout_seconds=0,
            clock=clock,
        )
        recorded = _RecordingSocket(clock)

        send_frame(recorded, FrameType.BEGIN, 1, emulator=emulator)
        with self.assertRaisesRegex(LinkEmulatorError, "bounded"):
            send_frame(recorded, FrameType.BEGIN, 2, emulator=emulator)
        self.assertEqual(emulator.pending_frames, 1)

        _advance_and_wake(clock, emulator, 1.0)
        recorded.wait_for_calls(1)
        emulator.close(timeout_seconds=1)
        with self.assertRaisesRegex(LinkEmulatorError, "closed"):
            send_frame(recorded, FrameType.BEGIN, 3, emulator=emulator)
        self.assertIsNone(emulator._worker)

    def test_background_socket_error_is_propagated_and_wakes_flush(self) -> None:
        clock = _ManualClock()
        emulator = LinkEmulator(one_way_delay_ms=1_000, clock=clock)
        failing = _RecordingSocket(clock, fail=True)

        send_frame(failing, FrameType.BEGIN, 1, emulator=emulator)
        _advance_and_wake(clock, emulator, 1.0)
        failing.wait_for_calls(1)
        with self.assertRaisesRegex(LinkEmulatorError, "sender failed") as raised:
            emulator.flush(timeout_seconds=1)
        self.assertIsInstance(raised.exception.__cause__, OSError)
        self.assertEqual(failing.shutdown_calls, 1)

        # The asynchronous failure is sticky and is surfaced on every later send.
        with self.assertRaisesRegex(LinkEmulatorError, "sender failed"):
            send_frame(failing, FrameType.BEGIN, 2, emulator=emulator)


if __name__ == "__main__":
    unittest.main()
