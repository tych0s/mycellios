from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import sys


REQUEST = struct.Struct("<IIIII")
RESPONSE = struct.Struct("<II")
U32 = struct.Struct("<I")


def read_exact(size: int) -> bytes | None:
    data = bytearray()
    while len(data) < size:
        chunk = sys.stdin.buffer.read(size - len(data))
        if not chunk:
            return None if not data else bytes(data)
        data.extend(chunk)
    return bytes(data)


def write_response(status: int, payload: bytes, fragment: int) -> None:
    wire = RESPONSE.pack(status, len(payload)) + payload
    for offset in range(0, len(wire), fragment):
        sys.stdout.buffer.write(wire[offset : offset + fragment])
        sys.stdout.buffer.flush()


def log_event(path: Path | None, value: dict[str, object]) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"pid": os.getpid(), **value}, sort_keys=True) + "\n")


def token_for(values: tuple[float, ...], vocab_size: int) -> int:
    return int(abs(round(sum(values) * 1000))) % vocab_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--fake-hidden-size", type=int, required=True)
    parser.add_argument("--fake-vocab-size", type=int, required=True)
    parser.add_argument("--fake-event-log", type=Path)
    parser.add_argument("--fake-fragment-bytes", type=int, default=3)
    parser.add_argument(
        "--fake-version-sha",
        default="0c16119713396ec6052400f3eb049c5e7a66cd94",
    )
    parser.add_argument("sub_gguf", type=Path)
    parser.add_argument("mode", choices=("first", "middle", "last"))
    parser.add_argument("n_ctx", type=int)
    parser.add_argument("n_threads", type=int)
    parser.add_argument("n_gpu_layers", type=int)
    return parser.parse_args()


def main() -> int:
    if "--version" in sys.argv[1:]:
        version_sha = "0c16119713396ec6052400f3eb049c5e7a66cd94"
        if "--fake-version-sha" in sys.argv[1:]:
            index = sys.argv.index("--fake-version-sha")
            version_sha = sys.argv[index + 1]
        print("nakshatra-fabric-worker")
        print(f"  sha        {version_sha}")
        print("  built_on   gdlp-fake-daemon")
        return 0
    args = parse_args()
    if args.sub_gguf.read_bytes()[:4] != b"GGUF":
        return 2
    if args.fake_hidden_size < 1 or args.fake_vocab_size < 1:
        return 2
    fragment = max(1, args.fake_fragment_bytes)
    sequence_length = 0
    log_event(
        args.fake_event_log,
        {
            "event": "start",
            "mode": args.mode,
            "context": args.n_ctx,
            "threads": args.n_threads,
            "gpuLayers": args.n_gpu_layers,
        },
    )
    while True:
        header = read_exact(REQUEST.size)
        if header is None:
            break
        if len(header) != REQUEST.size:
            return 3
        command, n_tokens, start_pos, flags, payload_bytes = REQUEST.unpack(header)
        payload = read_exact(payload_bytes)
        if payload is None or len(payload) != payload_bytes:
            return 3
        before = sequence_length
        status = 0
        response = b""
        if command == 3:
            # Match the pinned daemon's known limitation: partial range and
            # endpoint flags are not surfaced reliably by INFO.
            response = struct.pack(
                "<6i", 0, 99, args.fake_hidden_size, 1, 1, args.fake_vocab_size
            )
        elif command == 4:
            if n_tokens != 0 or payload_bytes != 4:
                status = 2
            else:
                keep = U32.unpack(payload)[0]
                if keep > sequence_length:
                    status = 2
                else:
                    sequence_length = keep
        elif command == 2:
            expected = n_tokens * args.fake_hidden_size * 4
            if n_tokens < 1 or payload_bytes != expected:
                status = 2
            elif start_pos > sequence_length or start_pos + n_tokens > args.n_ctx:
                status = 1
            else:
                keep_kv = bool(flags & 0x1)
                all_logits = bool(flags & 0x2)
                if keep_kv:
                    sequence_length = start_pos
                else:
                    if start_pos != 0:
                        status = 2
                    sequence_length = 0
                if status == 0:
                    values = struct.unpack(f"<{n_tokens * args.fake_hidden_size}f", payload)
                    sequence_length += n_tokens
                    if args.mode == "last":
                        rows = tuple(
                            values[
                                index * args.fake_hidden_size :
                                (index + 1) * args.fake_hidden_size
                            ]
                            for index in range(n_tokens)
                        )
                        if all_logits:
                            tokens = tuple(
                                token_for(row, args.fake_vocab_size) for row in rows
                            )
                            response = U32.pack(2) + struct.pack(
                                f"<{n_tokens}i", *tokens
                            )
                        else:
                            response = U32.pack(1) + struct.pack(
                                "<i", token_for(rows[-1], args.fake_vocab_size)
                            )
                    else:
                        transformed = tuple(value + 0.25 for value in values)
                        response = U32.pack(0) + struct.pack(
                            f"<{len(transformed)}f", *transformed
                        )
        else:
            status = 2
        log_event(
            args.fake_event_log,
            {
                "event": "command",
                "command": command,
                "nTokens": n_tokens,
                "startPos": start_pos,
                "flags": flags,
                "sequenceBefore": before,
                "sequenceAfter": sequence_length,
                "status": status,
            },
        )
        write_response(status, response, fragment)
    log_event(args.fake_event_log, {"event": "eof", "sequence": sequence_length})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
