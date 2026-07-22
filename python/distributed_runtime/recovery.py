from __future__ import annotations

from collections.abc import Callable, Iterable
import copy
from concurrent.futures import (
    CancelledError as FutureCancelledError,
    FIRST_COMPLETED,
    Future,
    wait,
)
from dataclasses import dataclass, field
import threading
import time
from typing import Any

import torch

from .engine import (
    DistributedPipelineEngine,
    GenerationCancelledError,
    GenerationInput,
    GenerationOutput,
    PipelineRecoveryIdentity,
    TokenCallback,
)


EngineFactory = Callable[[], DistributedPipelineEngine]


class PipelineRecoveryError(RuntimeError):
    """A failed route could not be resumed without weakening exactness."""


class RecoveryCompatibilityError(PipelineRecoveryError):
    """A proposed standby does not run the identical immutable execution plan."""


@dataclass
class _ReplayJob:
    request: GenerationInput
    callback: TokenCallback | None
    future: Future[GenerationOutput] = field(default_factory=Future)
    lock: threading.Lock = field(default_factory=threading.Lock)
    cancel_requested: threading.Event = field(default_factory=threading.Event)
    emitted: list[int] = field(default_factory=list)
    arrivals: list[float] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)
    retries: int = 0
    active_attempt: int = 0
    active_attempt_base: int = 0
    callback_error: BaseException | None = None


class RecoveringPipelineEngine:
    """Token-exact greedy recovery around ``DistributedPipelineEngine``.

    Recovery never transfers a potentially partial KV cache.  It recreates an
    immutable-compatible route and prefills ``prompt + visible output tokens``;
    the next target-model argmax is therefore the exact continuation and is not
    emitted twice.  This contract is intentionally limited to deterministic
    greedy generation.  A future stochastic sampler must additionally capture
    and restore its complete RNG/sampler state before using this wrapper.
    """

    def __init__(
        self,
        engine_factory: EngineFactory,
        *,
        max_retries: int = 1,
        initial_engine: DistributedPipelineEngine | None = None,
        standby_factories: Iterable[EngineFactory] = (),
    ) -> None:
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if not callable(engine_factory):
            raise TypeError("engine_factory must be callable")
        standbys = tuple(standby_factories)
        if any(not callable(factory) for factory in standbys):
            raise TypeError("standby factories must be callable")

        engine = initial_engine if initial_engine is not None else engine_factory()
        if not engine.healthy:
            engine.close()
            raise PipelineRecoveryError("initial pipeline route is not healthy")

        if (
            not getattr(engine.config, "spawn_local_stages", True)
            and not getattr(engine.config, "stage_executor_ids", None)
        ):
            engine.close()
            raise RecoveryCompatibilityError(
                "remote recovery requires explicit sealed stage_executor_ids"
            )
        try:
            expected_identity = engine.recovery_identity
        except BaseException as error:
            engine.close()
            raise RecoveryCompatibilityError(
                f"pipeline route has no usable recovery execution contract: {error}"
            ) from error

        self.config = engine.config
        self.maximum_context = engine.maximum_context
        self.total_layers = engine.total_layers
        self.hidden_size = engine.hidden_size
        self.model_snapshot = engine.model_snapshot
        self.model_artifact = engine.model_artifact
        self.pipeline_id = engine.pipeline_id
        self._expected_identity = expected_identity
        self._root_parameter_bytes = engine.root_parameter_bytes
        self._engine_factory = engine_factory
        # Prefer a pre-provisioned standby after a fault; the primary factory is
        # always the final fallback and recreates the original route.
        self._replacement_factories = (*standbys, engine_factory)
        self._factory_cursor = 0
        self.max_retries = max_retries

        self._condition = threading.Condition(threading.RLock())
        self._engine = engine
        self._epoch = 0
        self._recovering = False
        self._closed = False
        self._terminal_error: BaseException | None = None
        self._jobs_by_client: dict[int, _ReplayJob] = {}
        self._workers: set[threading.Thread] = set()
        self._retired_stage_metrics: list[dict[str, Any]] = []

        self._route_recovery_attempts = 0
        self._route_recovery_successes = 0
        self._route_recovery_failures = 0
        self._compatibility_rejections = 0
        self._request_retry_attempts = 0
        self._recovered_requests = 0
        self._replayed_output_tokens = 0
        self._replayed_prompt_tokens = 0
        self._last_recovery_error: str | None = None
        self._last_recovery_duration_ms: float | None = None

    @property
    def stages(self) -> int:
        return len(self.config.boundaries) - 1

    @property
    def root_parameter_bytes(self) -> int:
        return self._root_parameter_bytes

    @property
    def execution_topology(self) -> dict[str, Any]:
        with self._condition:
            return copy.deepcopy(self._engine.execution_topology)

    @property
    def recovery_identity(self) -> PipelineRecoveryIdentity:
        return self._expected_identity

    @property
    def healthy(self) -> bool:
        with self._condition:
            if self._closed or self._terminal_error is not None or self._recovering:
                return False
            return self._engine.healthy

    @property
    def fatal_error(self) -> str | None:
        with self._condition:
            if self._terminal_error is not None:
                return str(self._terminal_error)
            if self._recovering:
                return None
            return self._engine.fatal_error

    @property
    def speculation_stats(self) -> dict[str, Any]:
        with self._condition:
            return self._engine.speculation_stats

    @property
    def root_batch_stats(self) -> dict[str, int | float]:
        with self._condition:
            return self._engine.root_batch_stats

    @property
    def prefill_window_stats(self) -> dict[str, int]:
        with self._condition:
            return self._engine.prefill_window_stats

    @property
    def stage_metrics(self) -> list[dict[str, Any]]:
        with self._condition:
            return [*self._retired_stage_metrics, *self._engine.stage_metrics]

    @property
    def recovery_stats(self) -> dict[str, Any]:
        with self._condition:
            if self._closed:
                state = "closed"
            elif self._terminal_error is not None:
                state = "degraded"
            elif self._recovering:
                state = "recovering"
            elif self._engine.healthy:
                state = "ready"
            else:
                state = "route-failed"
            return {
                "configured": True,
                "mode": "greedy-prefix-recompute",
                "state": state,
                "max_retries_per_request": self.max_retries,
                "route_epoch": self._epoch,
                "route_recovery_attempts": self._route_recovery_attempts,
                "route_recovery_successes": self._route_recovery_successes,
                "route_recovery_failures": self._route_recovery_failures,
                "compatibility_rejections": self._compatibility_rejections,
                "request_retry_attempts": self._request_retry_attempts,
                "recovered_requests": self._recovered_requests,
                "replayed_output_tokens": self._replayed_output_tokens,
                "replayed_prompt_tokens": self._replayed_prompt_tokens,
                "active_requests": len(self._jobs_by_client),
                "last_recovery_error": self._last_recovery_error,
                "last_recovery_duration_ms": self._last_recovery_duration_ms,
            }

    def generate(
        self,
        requests: list[GenerationInput],
        on_token: TokenCallback | None = None,
    ) -> list[GenerationOutput]:
        futures = self.submit(requests, on_token)
        outputs: list[GenerationOutput] = []
        first_cancelled: GenerationCancelledError | None = None
        for request, future in zip(requests, futures):
            try:
                outputs.append(future.result())
            except FutureCancelledError:
                first_cancelled = first_cancelled or GenerationCancelledError(
                    f"generation {request.client_id} was cancelled"
                )
        if first_cancelled is not None:
            raise first_cancelled
        return outputs

    def submit(
        self,
        requests: list[GenerationInput],
        on_token: TokenCallback | None = None,
    ) -> list[Future[GenerationOutput]]:
        if not requests:
            return []
        if len({request.client_id for request in requests}) != len(requests):
            raise ValueError("client_id values must be unique within a batch")

        jobs = [
            _ReplayJob(
                request=GenerationInput(
                    client_id=request.client_id,
                    input_ids=request.input_ids.detach().to(device="cpu").clone(),
                    max_new_tokens=request.max_new_tokens,
                    eos_token_ids=request.eos_token_ids,
                ),
                callback=on_token,
            )
            for request in requests
        ]
        with self._condition:
            self._raise_unavailable_locked()
            duplicate = next(
                (
                    job.request.client_id
                    for job in jobs
                    if job.request.client_id in self._jobs_by_client
                ),
                None,
            )
            if duplicate is not None:
                raise ValueError(f"client_id already active: {duplicate}")
            if len(self._jobs_by_client) + len(jobs) > self.config.max_pending_requests:
                raise RuntimeError("pipeline request queue is full")
            for job in jobs:
                self._jobs_by_client[job.request.client_id] = job

            worker = threading.Thread(
                target=self._run_group,
                args=(jobs,),
                name=f"pipeline-replay-{jobs[0].request.client_id}",
                daemon=True,
            )
            self._workers.add(worker)
            worker.start()
        return [job.future for job in jobs]

    def cancel(self, client_id: int) -> bool:
        with self._condition:
            job = self._jobs_by_client.get(client_id)
            if job is None:
                return False
            job.cancel_requested.set()
            engine = self._engine
        engine.cancel(client_id)
        return True

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            jobs = tuple(self._jobs_by_client.values())
            workers = tuple(self._workers)
            engine = self._engine
            for job in jobs:
                job.cancel_requested.set()
            self._condition.notify_all()
        engine.close()
        for worker in workers:
            if worker is not threading.current_thread():
                worker.join(timeout=5.0)
        with self._condition:
            self._retired_stage_metrics.extend(engine.stage_metrics)

    def __enter__(self) -> "RecoveringPipelineEngine":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _run_group(self, jobs: list[_ReplayJob]) -> None:
        pending = list(jobs)
        try:
            while pending:
                pending = [job for job in pending if not self._retire_if_cancelled(job)]
                pending = [job for job in pending if not self._retire_if_prefix_complete(job)]
                if not pending:
                    return

                try:
                    engine, epoch = self._route_snapshot()
                except BaseException as error:
                    for job in pending:
                        self._fail_job(job, error)
                    return

                attempt_inputs: list[GenerationInput] = []
                attempt_serials: dict[int, int] = {}
                for job in pending:
                    with job.lock:
                        job.active_attempt += 1
                        job.active_attempt_base = len(job.emitted)
                        job.callback_error = None
                        attempt_inputs.append(self._replay_input_locked(job))
                        attempt = job.active_attempt
                        attempt_serials[job.request.client_id] = attempt
                    # Every job advances its attempt serial together in this group.
                    if attempt < 1:
                        raise AssertionError("attempt serial must be positive")
                attempt_by_id = {job.request.client_id: job for job in pending}

                def on_attempt_token(
                    client_id: int,
                    token_id: int,
                    step: int,
                    _arrived: float,
                ) -> None:
                    job = attempt_by_id.get(client_id)
                    if job is None:
                        return
                    callback: TokenCallback | None = None
                    absolute_step = 0
                    observed_at = time.perf_counter()
                    with job.lock:
                        if (
                            job.active_attempt != attempt_serials[client_id]
                            or job.cancel_requested.is_set()
                            or job.future.done()
                        ):
                            return
                        expected_step = len(job.emitted) - job.active_attempt_base
                        if step != expected_step:
                            job.callback_error = PipelineRecoveryError(
                                f"request {client_id} callback step {step} does not match "
                                f"replay step {expected_step}"
                            )
                            return
                        absolute_step = len(job.emitted)
                        job.emitted.append(int(token_id))
                        job.arrivals.append(observed_at)
                        callback = job.callback
                    if callback is not None:
                        try:
                            callback(client_id, int(token_id), absolute_step, observed_at)
                        except BaseException:
                            pass

                try:
                    attempt_futures = engine.submit(attempt_inputs, on_attempt_token)
                except BaseException as error:
                    self._retry_or_fail(pending, engine, epoch, error)
                    pending = [job for job in pending if not job.future.done()]
                    continue

                outstanding = {
                    future: job for future, job in zip(attempt_futures, pending)
                }
                failed: list[tuple[_ReplayJob, BaseException]] = []
                while outstanding:
                    done, _ = wait(
                        tuple(outstanding),
                        timeout=0.05,
                        return_when=FIRST_COMPLETED,
                    )
                    if not done:
                        # A bare Future transitions to CANCELLED (not the
                        # executor-only CANCELLED_AND_NOTIFIED state), which
                        # concurrent.futures.wait does not classify as done.
                        # Polling also makes cancellation independent of a
                        # wedged transport acknowledging CANCEL.
                        for future, job in tuple(outstanding.items()):
                            if future.cancelled() or job.cancel_requested.is_set():
                                outstanding.pop(future)
                                self._cancel_job(job)
                        continue
                    for future in done:
                        job = outstanding.pop(future)
                        try:
                            result = future.result()
                            self._finish_attempt(job, result)
                        except FutureCancelledError as error:
                            if job.cancel_requested.is_set():
                                self._cancel_job(job)
                            else:
                                failed.append((job, error))
                        except BaseException as error:
                            failed.append((job, error))

                if not failed:
                    return
                failed_jobs = [job for job, _ in failed if not job.future.done()]
                if not failed_jobs:
                    return
                first_error = failed[0][1]
                self._retry_or_fail(failed_jobs, engine, epoch, first_error)
                pending = [job for job in failed_jobs if not job.future.done()]
        except BaseException as error:
            for job in pending:
                self._fail_job(job, error)
        finally:
            with self._condition:
                self._workers.discard(threading.current_thread())

    def _retry_or_fail(
        self,
        jobs: list[_ReplayJob],
        engine: DistributedPipelineEngine,
        epoch: int,
        error: BaseException,
    ) -> None:
        if not self._route_failure_is_recoverable(engine):
            for job in jobs:
                self._fail_job(job, error)
            return

        retryable: list[_ReplayJob] = []
        for job in jobs:
            if self._retire_if_cancelled(job) or self._retire_if_prefix_complete(job):
                continue
            with job.lock:
                job.retries += 1
                retries = job.retries
            with self._condition:
                self._request_retry_attempts += 1
            if retries > self.max_retries:
                self._fail_job(
                    job,
                    PipelineRecoveryError(
                        f"request {job.request.client_id} exceeded max_retries="
                        f"{self.max_retries} after route failure: {error}"
                    ),
                )
            else:
                retryable.append(job)
        if not retryable:
            return
        try:
            self._recover_route(engine, epoch, error)
        except BaseException as recovery_error:
            for job in retryable:
                self._fail_job(job, recovery_error)

    def _recover_route(
        self,
        failed_engine: DistributedPipelineEngine,
        failed_epoch: int,
        cause: BaseException,
    ) -> None:
        with self._condition:
            while self._recovering and self._epoch == failed_epoch and not self._closed:
                self._condition.wait()
            self._raise_unavailable_locked()
            if self._epoch != failed_epoch:
                return
            if self._engine is not failed_engine:
                return
            self._recovering = True
            self._last_recovery_error = str(cause)

        started = time.perf_counter()
        try:
            failed_engine.close()
            with self._condition:
                self._retired_stage_metrics.extend(failed_engine.stage_metrics)

            candidate: DistributedPipelineEngine | None = None
            failures: list[str] = []
            factory_count = len(self._replacement_factories)
            for offset in range(factory_count):
                with self._condition:
                    index = (self._factory_cursor + offset) % factory_count
                    self._route_recovery_attempts += 1
                factory = self._replacement_factories[index]
                proposed: DistributedPipelineEngine | None = None
                try:
                    proposed = factory()
                    self._validate_replacement(proposed)
                except BaseException as error:
                    failures.append(f"factory[{index}]: {error}")
                    try:
                        if proposed is not None:
                            proposed.close()
                    except AttributeError:
                        pass
                    continue
                if proposed is None:
                    raise AssertionError("recovery factory returned no engine")
                candidate = proposed
                with self._condition:
                    self._factory_cursor = (index + 1) % factory_count
                break

            if candidate is None:
                raise PipelineRecoveryError(
                    "no immutable-compatible recovery route was available: "
                    + "; ".join(failures)
                )

            with self._condition:
                if self._closed:
                    candidate.close()
                    raise PipelineRecoveryError("pipeline recovery was closed")
                self._engine = candidate
                self._epoch += 1
                self._route_recovery_successes += 1
        except BaseException as error:
            with self._condition:
                self._route_recovery_failures += 1
                self._terminal_error = error
            raise
        finally:
            with self._condition:
                self._last_recovery_duration_ms = (
                    time.perf_counter() - started
                ) * 1_000
                self._recovering = False
                self._condition.notify_all()

    def _validate_replacement(self, engine: DistributedPipelineEngine) -> None:
        if not engine.healthy:
            raise PipelineRecoveryError("replacement pipeline route is not healthy")
        actual = engine.recovery_identity
        if actual != self._expected_identity:
            with self._condition:
                self._compatibility_rejections += 1
            differences = [
                name
                for name in PipelineRecoveryIdentity.__dataclass_fields__
                if getattr(actual, name) != getattr(self._expected_identity, name)
            ]
            raise RecoveryCompatibilityError(
                "replacement pipeline recovery identity differs in: "
                + ", ".join(differences)
            )

    def _route_snapshot(self) -> tuple[DistributedPipelineEngine, int]:
        with self._condition:
            while self._recovering and not self._closed:
                self._condition.wait()
            self._raise_unavailable_locked()
            return self._engine, self._epoch

    @staticmethod
    def _route_failure_is_recoverable(engine: DistributedPipelineEngine) -> bool:
        try:
            return not engine.healthy or engine.fatal_error is not None
        except BaseException:
            return True

    def _replay_input_locked(self, job: _ReplayJob) -> GenerationInput:
        original = job.request.input_ids
        if job.emitted:
            replayed = torch.tensor(
                [job.emitted],
                dtype=original.dtype,
                device="cpu",
            )
            input_ids = torch.cat((original, replayed), dim=1)
            with self._condition:
                self._replayed_output_tokens += len(job.emitted)
                self._replayed_prompt_tokens += int(original.shape[1])
        else:
            input_ids = original
        return GenerationInput(
            client_id=job.request.client_id,
            input_ids=input_ids,
            max_new_tokens=job.request.max_new_tokens - len(job.emitted),
            eos_token_ids=job.request.eos_token_ids,
        )

    def _finish_attempt(self, job: _ReplayJob, result: GenerationOutput) -> None:
        with job.lock:
            if job.callback_error is not None:
                raise job.callback_error
            suffix = tuple(job.emitted[job.active_attempt_base :])
            if suffix != result.token_ids:
                raise PipelineRecoveryError(
                    f"request {job.request.client_id} callback/result token mismatch"
                )
            finish_reason = result.finish_reason
        self._complete_job(job, finish_reason)

    def _retire_if_prefix_complete(self, job: _ReplayJob) -> bool:
        with job.lock:
            if not job.emitted:
                return False
            if job.emitted[-1] in job.request.eos_token_ids:
                reason = "stop"
            elif len(job.emitted) >= job.request.max_new_tokens:
                reason = "length"
            else:
                return False
        self._complete_job(job, reason)
        return True

    def _retire_if_cancelled(self, job: _ReplayJob) -> bool:
        if not job.cancel_requested.is_set():
            return False
        self._cancel_job(job)
        return True

    def _complete_job(self, job: _ReplayJob, finish_reason: str) -> None:
        with job.lock:
            if job.future.done():
                return
            intervals = [
                (right - left) * 1_000
                for left, right in zip(job.arrivals, job.arrivals[1:])
            ]
            output = GenerationOutput(
                client_id=job.request.client_id,
                token_ids=tuple(job.emitted),
                finish_reason=finish_reason,
                ttft_ms=(job.arrivals[0] - job.started_at) * 1_000,
                tpot_ms=sum(intervals) / len(intervals) if intervals else 0.0,
                total_ms=(job.arrivals[-1] - job.started_at) * 1_000,
            )
            recovered = job.retries > 0
            job.future.set_result(output)
        with self._condition:
            self._jobs_by_client.pop(job.request.client_id, None)
            if recovered:
                self._recovered_requests += 1

    def _cancel_job(self, job: _ReplayJob) -> None:
        with job.lock:
            if not job.future.done():
                job.future.cancel()
        with self._condition:
            self._jobs_by_client.pop(job.request.client_id, None)

    def _fail_job(self, job: _ReplayJob, error: BaseException) -> None:
        with job.lock:
            if job.future.done():
                return
            job.future.set_exception(error)
        with self._condition:
            self._jobs_by_client.pop(job.request.client_id, None)

    def _raise_unavailable_locked(self) -> None:
        if self._closed:
            raise RuntimeError("pipeline recovery engine is closed")
        if self._terminal_error is not None:
            raise PipelineRecoveryError(
                f"pipeline recovery is degraded: {self._terminal_error}"
            ) from self._terminal_error
