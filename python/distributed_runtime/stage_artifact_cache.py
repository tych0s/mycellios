from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from typing import BinaryIO, Iterator, Mapping, Sequence
import urllib.error
import urllib.parse
import urllib.request

from .stage_artifact import (
    STAGE_ARTIFACT_MANIFEST,
    SafeTensorsStageArtifact,
    parse_stage_artifact_manifest,
    verify_stage_artifact,
)


STAGE_CACHE_PARTIAL_SCHEMA = "mycellios-stage-cache-partial/1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CONTENT_RANGE_PATTERN = re.compile(r"^bytes ([0-9]+)-([0-9]+)/([0-9]+)$")


class StageArtifactCacheError(RuntimeError):
    """Base error for native stage cache acquisition and maintenance."""


class StageArtifactTransportError(StageArtifactCacheError):
    """A source stopped before a declared payload was completely transferred."""


class StageArtifactIntegrityError(StageArtifactCacheError):
    """A source, partial, or immutable cache object violated its sealed contract."""


@dataclass(frozen=True, slots=True)
class StageArtifactCacheLimits:
    max_manifest_bytes: int = 4 * 1024 * 1024
    max_payload_bytes: int = 256 * 1024 * 1024 * 1024
    max_package_bytes: int = 512 * 1024 * 1024 * 1024
    transfer_chunk_bytes: int = 1024 * 1024
    http_timeout_seconds: float = 30.0
    lock_timeout_seconds: float = 600.0

    def __post_init__(self) -> None:
        for name in (
            "max_manifest_bytes",
            "max_payload_bytes",
            "max_package_bytes",
            "transfer_chunk_bytes",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("http_timeout_seconds", "lock_timeout_seconds"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive")
        if self.max_payload_bytes > self.max_package_bytes:
            raise ValueError("max_payload_bytes cannot exceed max_package_bytes")


@dataclass(frozen=True, slots=True)
class CachedStageObject:
    path: str
    role: str
    sha256: str
    size_bytes: int
    downloaded_bytes: int
    resumed_from_bytes: int
    reused: bool


@dataclass(frozen=True, slots=True)
class StageArtifactAcquisition:
    cache_root: str
    package_directory: str
    package_id: str
    artifact_identity: str
    manifest_sha256: str
    downloaded_bytes: int
    resumed_bytes: int
    materialized: bool
    objects: tuple[CachedStageObject, ...]


@dataclass(frozen=True, slots=True)
class StageArtifactCleanup:
    cache_root: str
    target_kind: str
    target: str
    package_removed: bool
    objects_removed: tuple[str, ...]
    partials_removed: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ManifestOrigin:
    kind: str
    locator: str
    base: str

    def payload_locator(self, name: str) -> str:
        if self.kind == "http":
            return urllib.parse.urljoin(self.base, name)
        return str((Path(self.base) / name).resolve())


@dataclass(slots=True)
class _PartialState:
    digest: str
    expected_size: int
    source: str
    version: str | None
    etag: str | None
    confirmed_offset: int

    def to_document(self) -> dict[str, object]:
        return {
            "schema": STAGE_CACHE_PARTIAL_SCHEMA,
            "digest": self.digest,
            "expectedSizeBytes": self.expected_size,
            "source": self.source,
            "version": self.version,
            "etag": self.etag,
            "confirmedOffset": self.confirmed_offset,
        }


class StageArtifactCache:
    """Native content-addressed cache for sealed Mycellios stage packages.

    Payloads are streamed into digest-specific ``.part`` files. The confirmed
    offset is advanced only after payload bytes have reached durable storage.
    An object becomes visible at its SHA-256 path only after its exact declared
    size and digest have been verified.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        limits: StageArtifactCacheLimits | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.limits = limits or StageArtifactCacheLimits()
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir() or self.root.is_symlink():
            raise StageArtifactCacheError(
                f"cache root is not a regular directory: {self.root}"
            )
        for child in ("objects", "partials", "locks", "packages"):
            (self.root / child).mkdir(parents=True, exist_ok=True)

    def object_path(self, digest: str) -> Path:
        digest = _digest(digest, "object digest")
        return self.root / "objects" / "sha256" / digest[:2] / digest

    def acquire(
        self,
        manifest_source: str | os.PathLike[str],
        *,
        destination: str | os.PathLike[str] | None = None,
        expected_package_id: str | None = None,
    ) -> StageArtifactAcquisition:
        manifest_bytes, origin = self._read_manifest(manifest_source)
        manifest = _parse_canonical_manifest(manifest_bytes)
        package_id = manifest.package_id
        if expected_package_id is not None:
            expected = _digest(expected_package_id, "expected package ID")
            if expected != package_id:
                raise StageArtifactIntegrityError(
                    "stage package identity does not match the requested package"
                )

        document = manifest.to_document()
        payload_size = sum(int(item["sizeBytes"]) for item in document["files"])
        if payload_size > self.limits.max_package_bytes:
            raise StageArtifactIntegrityError(
                "stage package exceeds the configured package size limit"
            )
        for item in document["files"]:
            if int(item["sizeBytes"]) > self.limits.max_payload_bytes:
                raise StageArtifactIntegrityError(
                    f"stage payload exceeds the configured file size limit: "
                    f"{item['path']}"
                )

        objects: list[CachedStageObject] = []
        manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
        objects.append(
            self._cache_embedded_bytes(
                manifest_bytes,
                digest=manifest_digest,
                role="manifest",
                logical_path=STAGE_ARTIFACT_MANIFEST,
            )
        )
        for item in document["files"]:
            objects.append(
                self._acquire_payload(
                    logical_path=str(item["path"]),
                    role=str(item["role"]),
                    digest=str(item["sha256"]),
                    expected_size=int(item["sizeBytes"]),
                    source=origin.payload_locator(str(item["path"])),
                    source_kind=origin.kind,
                )
            )

        package_directory = (
            Path(destination).resolve()
            if destination is not None
            else self.root / "packages" / package_id
        )
        materialized = self._materialize_package(
            package_directory,
            package_id=package_id,
            manifest_digest=manifest_digest,
            files=document["files"],
        )
        return StageArtifactAcquisition(
            cache_root=str(self.root),
            package_directory=str(package_directory),
            package_id=package_id,
            artifact_identity=f"sha256:{package_id}",
            manifest_sha256=manifest_digest,
            downloaded_bytes=sum(item.downloaded_bytes for item in objects),
            resumed_bytes=sum(item.resumed_from_bytes for item in objects),
            materialized=materialized,
            objects=tuple(objects),
        )

    def cleanup(
        self,
        *,
        package_id: str | None = None,
        digest: str | None = None,
        prune_package_objects: bool = False,
    ) -> StageArtifactCleanup:
        if (package_id is None) == (digest is None):
            raise ValueError("select exactly one package_id or digest to clean")
        if package_id is not None:
            return self._cleanup_package(
                _digest(package_id, "package ID"),
                prune_objects=prune_package_objects,
            )
        assert digest is not None
        object_digest = _digest(digest, "object digest")
        if prune_package_objects:
            raise ValueError("prune_package_objects is valid only with a package ID")
        with self._catalog_lock():
            if self._object_is_referenced(object_digest):
                raise StageArtifactCacheError(
                    f"cache object is referenced by a materialized package: "
                    f"{object_digest}"
                )
            object_removed, partials = self._remove_object_and_partial(object_digest)
        return StageArtifactCleanup(
            cache_root=str(self.root),
            target_kind="digest",
            target=object_digest,
            package_removed=False,
            objects_removed=(object_digest,) if object_removed else (),
            partials_removed=partials,
        )

    def _read_manifest(
        self, source: str | os.PathLike[str]
    ) -> tuple[bytes, _ManifestOrigin]:
        text = os.fspath(source)
        parsed = urllib.parse.urlparse(text)
        if parsed.scheme in {"http", "https"}:
            request = urllib.request.Request(
                text,
                headers={"Accept-Encoding": "identity"},
                method="GET",
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.limits.http_timeout_seconds
                ) as response:
                    status = response.getcode()
                    if status != 200:
                        raise StageArtifactTransportError(
                            f"manifest source returned HTTP {status}"
                        )
                    _reject_encoded_response(response.headers)
                    declared = _optional_content_length(response.headers)
                    if (
                        declared is not None
                        and declared > self.limits.max_manifest_bytes
                    ):
                        raise StageArtifactIntegrityError(
                            "stage manifest exceeds the configured size limit"
                        )
                    value = response.read(self.limits.max_manifest_bytes + 1)
                    if len(value) > self.limits.max_manifest_bytes:
                        raise StageArtifactIntegrityError(
                            "stage manifest exceeds the configured size limit"
                        )
                    if declared is not None and len(value) != declared:
                        raise StageArtifactTransportError(
                            "manifest source stopped before Content-Length"
                        )
                    effective_url = response.geturl()
            except StageArtifactCacheError:
                raise
            except (
                OSError,
                http.client.HTTPException,
                urllib.error.URLError,
            ) as error:
                raise StageArtifactTransportError(
                    f"could not download stage manifest: {error}"
                ) from error
            return value, _ManifestOrigin(
                kind="http",
                locator=effective_url,
                base=urllib.parse.urljoin(effective_url, "./"),
            )
        if parsed.scheme == "file":
            path = Path(urllib.request.url2pathname(parsed.path)).resolve()
        elif parsed.scheme and not (
            len(parsed.scheme) == 1 and len(text) >= 2 and text[1] == ":"
        ):
            raise ValueError("manifest source must be a local path or HTTP(S) URL")
        else:
            path = Path(text).resolve()
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"stage manifest does not exist: {path}")
        size = path.stat().st_size
        if size > self.limits.max_manifest_bytes:
            raise StageArtifactIntegrityError(
                "stage manifest exceeds the configured size limit"
            )
        value = path.read_bytes()
        if len(value) != size:
            raise StageArtifactTransportError(
                "local stage manifest changed while it was being read"
            )
        return value, _ManifestOrigin(
            kind="local",
            locator=str(path),
            base=str(path.parent),
        )

    def _cache_embedded_bytes(
        self,
        value: bytes,
        *,
        digest: str,
        role: str,
        logical_path: str,
    ) -> CachedStageObject:
        object_path = self.object_path(digest)
        lock_path = self._lock_path("object", digest)
        with _exclusive_file_lock(
            lock_path, timeout_seconds=self.limits.lock_timeout_seconds
        ):
            if object_path.exists():
                self._verify_existing_object(
                    object_path, digest=digest, expected_size=len(value)
                )
                return CachedStageObject(
                    path=logical_path,
                    role=role,
                    sha256=digest,
                    size_bytes=len(value),
                    downloaded_bytes=0,
                    resumed_from_bytes=0,
                    reused=True,
                )
            object_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = object_path.with_name(
                f".{object_path.name}.{os.getpid()}.{time.time_ns()}.part"
            )
            try:
                with temporary.open("xb") as handle:
                    handle.write(value)
                    handle.flush()
                    os.fsync(handle.fileno())
                if hashlib.sha256(value).hexdigest() != digest:
                    raise StageArtifactIntegrityError(
                        "embedded cache object digest changed before publication"
                    )
                if object_path.exists():
                    self._verify_existing_object(
                        object_path, digest=digest, expected_size=len(value)
                    )
                    temporary.unlink(missing_ok=True)
                else:
                    os.replace(temporary, object_path)
                    _fsync_directory(object_path.parent)
            finally:
                temporary.unlink(missing_ok=True)
        return CachedStageObject(
            path=logical_path,
            role=role,
            sha256=digest,
            size_bytes=len(value),
            downloaded_bytes=len(value),
            resumed_from_bytes=0,
            reused=False,
        )

    def _acquire_payload(
        self,
        *,
        logical_path: str,
        role: str,
        digest: str,
        expected_size: int,
        source: str,
        source_kind: str,
    ) -> CachedStageObject:
        digest = _digest(digest, "payload digest")
        object_path = self.object_path(digest)
        with _exclusive_file_lock(
            self._lock_path("object", digest),
            timeout_seconds=self.limits.lock_timeout_seconds,
        ):
            if object_path.exists():
                self._verify_existing_object(
                    object_path,
                    digest=digest,
                    expected_size=expected_size,
                )
                return CachedStageObject(
                    path=logical_path,
                    role=role,
                    sha256=digest,
                    size_bytes=expected_size,
                    downloaded_bytes=0,
                    resumed_from_bytes=0,
                    reused=True,
                )
            part_path, state_path = self._partial_paths(digest)
            state = self._load_or_create_partial(
                part_path,
                state_path,
                digest=digest,
                expected_size=expected_size,
                source=source,
            )
            resumed_from = state.confirmed_offset
            if source_kind == "http":
                downloaded = self._stream_http_payload(
                    source,
                    part_path=part_path,
                    state_path=state_path,
                    state=state,
                )
            elif source_kind == "local":
                downloaded = self._stream_local_payload(
                    Path(source),
                    part_path=part_path,
                    state_path=state_path,
                    state=state,
                )
            else:
                raise ValueError(f"unsupported payload source kind: {source_kind}")

            if state.confirmed_offset != expected_size:
                raise StageArtifactTransportError(
                    f"payload stopped at {state.confirmed_offset} of "
                    f"{expected_size} bytes"
                )
            actual_digest = _sha256_file(part_path, self.limits.transfer_chunk_bytes)
            if actual_digest != digest:
                part_path.unlink(missing_ok=True)
                state_path.unlink(missing_ok=True)
                raise StageArtifactIntegrityError(
                    f"stage payload SHA-256 mismatch for {logical_path}"
                )
            object_path.parent.mkdir(parents=True, exist_ok=True)
            if object_path.exists():
                self._verify_existing_object(
                    object_path,
                    digest=digest,
                    expected_size=expected_size,
                )
                part_path.unlink(missing_ok=True)
            else:
                os.replace(part_path, object_path)
                _fsync_directory(object_path.parent)
            state_path.unlink(missing_ok=True)
            _fsync_directory(state_path.parent)
            return CachedStageObject(
                path=logical_path,
                role=role,
                sha256=digest,
                size_bytes=expected_size,
                downloaded_bytes=downloaded,
                resumed_from_bytes=resumed_from,
                reused=False,
            )

    def _load_or_create_partial(
        self,
        part_path: Path,
        state_path: Path,
        *,
        digest: str,
        expected_size: int,
        source: str,
    ) -> _PartialState:
        part_path.parent.mkdir(parents=True, exist_ok=True)
        if state_path.exists():
            if state_path.is_symlink() or not state_path.is_file():
                raise StageArtifactIntegrityError(
                    f"partial state is not a regular file: {state_path}"
                )
            state = _parse_partial_state(state_path.read_bytes())
            if (
                state.digest != digest
                or state.expected_size != expected_size
                or state.source != source
            ):
                raise StageArtifactIntegrityError(
                    "partial state does not match the requested source contract"
                )
            if not part_path.exists() and state.confirmed_offset == 0:
                part_path.touch(exist_ok=False)
            elif not part_path.is_file() or part_path.is_symlink():
                raise StageArtifactIntegrityError(
                    f"partial payload is not a regular file: {part_path}"
                )
            current_size = part_path.stat().st_size
            if current_size < state.confirmed_offset:
                raise StageArtifactIntegrityError(
                    "partial payload is shorter than its confirmed offset"
                )
            if current_size > state.confirmed_offset:
                with part_path.open("r+b") as handle:
                    handle.truncate(state.confirmed_offset)
                    handle.flush()
                    os.fsync(handle.fileno())
            return state
        if part_path.exists():
            raise StageArtifactIntegrityError(
                "partial payload exists without confirmed offset state"
            )
        state = _PartialState(
            digest=digest,
            expected_size=expected_size,
            source=source,
            version=None,
            etag=None,
            confirmed_offset=0,
        )
        part_path.touch(exist_ok=False)
        self._write_partial_state(state_path, state)
        return state

    def _stream_local_payload(
        self,
        source: Path,
        *,
        part_path: Path,
        state_path: Path,
        state: _PartialState,
    ) -> int:
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"local stage payload does not exist: {source}")
        downloaded = 0
        with source.open("rb") as input_handle:
            initial_stat = os.fstat(input_handle.fileno())
            version = _local_source_version(initial_stat)
            if initial_stat.st_size != state.expected_size:
                raise StageArtifactIntegrityError(
                    "local stage payload size differs from its manifest"
                )
            if state.version is not None and state.version != version:
                raise StageArtifactIntegrityError(
                    "local stage payload changed since the partial was confirmed"
                )
            if state.version is None:
                state.version = version
                self._write_partial_state(state_path, state)
            input_handle.seek(state.confirmed_offset)
            with part_path.open("ab") as output_handle:
                while state.confirmed_offset < state.expected_size:
                    remaining = state.expected_size - state.confirmed_offset
                    chunk = input_handle.read(
                        min(self.limits.transfer_chunk_bytes, remaining)
                    )
                    if not chunk:
                        break
                    self._commit_partial_chunk(
                        output_handle,
                        state_path=state_path,
                        state=state,
                        chunk=chunk,
                    )
                    downloaded += len(chunk)
            final_stat = os.fstat(input_handle.fileno())
            if _local_source_version(final_stat) != version:
                raise StageArtifactIntegrityError(
                    "local stage payload changed during transfer"
                )
        return downloaded

    def _stream_http_payload(
        self,
        source: str,
        *,
        part_path: Path,
        state_path: Path,
        state: _PartialState,
    ) -> int:
        offset = state.confirmed_offset
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
            if state.etag is None:
                raise StageArtifactIntegrityError(
                    "HTTP partial has no ETag for a safe resume"
                )
            headers["If-Range"] = state.etag
        request = urllib.request.Request(source, headers=headers, method="GET")
        downloaded = 0
        try:
            with urllib.request.urlopen(
                request, timeout=self.limits.http_timeout_seconds
            ) as response:
                status = response.getcode()
                expected_status = 206 if offset else 200
                if status != expected_status:
                    if offset and status == 200:
                        raise StageArtifactIntegrityError(
                            "HTTP source ignored Range; partial was not modified"
                        )
                    raise StageArtifactTransportError(
                        f"stage payload source returned HTTP {status}, "
                        f"expected {expected_status}"
                    )
                _reject_encoded_response(response.headers)
                etag = _strong_etag(response.headers.get("ETag"))
                if state.etag is not None and etag != state.etag:
                    raise StageArtifactIntegrityError(
                        "HTTP source ETag changed; partial was not modified"
                    )
                if state.etag is None:
                    state.etag = etag
                    self._write_partial_state(state_path, state)

                remaining = state.expected_size - offset
                content_length = _optional_content_length(response.headers)
                if content_length is None or content_length != remaining:
                    raise StageArtifactIntegrityError(
                        "HTTP Content-Length does not match the sealed payload"
                    )
                if offset:
                    _validate_content_range(
                        response.headers.get("Content-Range"),
                        offset=offset,
                        expected_size=state.expected_size,
                    )
                with part_path.open("ab") as output_handle:
                    while state.confirmed_offset < state.expected_size:
                        remaining = state.expected_size - state.confirmed_offset
                        try:
                            chunk = response.read(
                                min(self.limits.transfer_chunk_bytes, remaining)
                            )
                        except http.client.IncompleteRead as error:
                            if error.partial:
                                self._commit_partial_chunk(
                                    output_handle,
                                    state_path=state_path,
                                    state=state,
                                    chunk=error.partial,
                                )
                                downloaded += len(error.partial)
                            raise StageArtifactTransportError(
                                "HTTP payload transport stopped mid-response"
                            ) from error
                        if not chunk:
                            break
                        self._commit_partial_chunk(
                            output_handle,
                            state_path=state_path,
                            state=state,
                            chunk=chunk,
                        )
                        downloaded += len(chunk)
        except StageArtifactCacheError:
            raise
        except (
            OSError,
            http.client.HTTPException,
            urllib.error.URLError,
        ) as error:
            raise StageArtifactTransportError(
                f"HTTP payload transport failed: {error}"
            ) from error
        if state.confirmed_offset != state.expected_size:
            raise StageArtifactTransportError(
                f"HTTP payload stopped at {state.confirmed_offset} of "
                f"{state.expected_size} bytes"
            )
        return downloaded

    def _commit_partial_chunk(
        self,
        handle: BinaryIO,
        *,
        state_path: Path,
        state: _PartialState,
        chunk: bytes,
    ) -> None:
        if not chunk:
            return
        new_offset = state.confirmed_offset + len(chunk)
        if new_offset > state.expected_size:
            raise StageArtifactIntegrityError(
                "source sent bytes beyond the sealed payload size"
            )
        handle.write(chunk)
        handle.flush()
        os.fsync(handle.fileno())
        state.confirmed_offset = new_offset
        self._write_partial_state(state_path, state)

    def _write_partial_state(self, state_path: Path, state: _PartialState) -> None:
        value = (
            json.dumps(
                state.to_document(),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        _write_bytes_atomic(state_path, value)

    def _materialize_package(
        self,
        destination: Path,
        *,
        package_id: str,
        manifest_digest: str,
        files: Sequence[Mapping[str, object]],
    ) -> bool:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination_key = hashlib.sha256(str(destination).encode("utf-8")).hexdigest()
        with _exclusive_file_lock(
            self._lock_path("materialize", destination_key),
            timeout_seconds=self.limits.lock_timeout_seconds,
        ):
            with self._catalog_lock():
                if destination.exists():
                    try:
                        verify_stage_artifact(
                            destination, expected_package_id=package_id
                        )
                    except Exception as error:
                        raise FileExistsError(
                            "stage destination already exists and is not the "
                            "requested sealed package"
                        ) from error
                    return False
                temporary = Path(
                    tempfile.mkdtemp(
                        prefix=f".{destination.name}.materialize-",
                        dir=destination.parent,
                    )
                )
                try:
                    entries = [
                        (
                            STAGE_ARTIFACT_MANIFEST,
                            self.object_path(manifest_digest),
                        ),
                        *(
                            (
                                str(item["path"]),
                                self.object_path(str(item["sha256"])),
                            )
                            for item in files
                        ),
                    ]
                    for name, object_path in entries:
                        target = temporary / name
                        self._link_or_stream_copy(object_path, target)
                    verify_stage_artifact(temporary, expected_package_id=package_id)
                    if destination.exists():
                        raise FileExistsError(
                            f"stage destination appeared during publication: "
                            f"{destination}"
                        )
                    os.rename(temporary, destination)
                    _fsync_directory(destination.parent)
                except BaseException:
                    shutil.rmtree(temporary, ignore_errors=True)
                    raise
        return True

    def _link_or_stream_copy(self, source: Path, destination: Path) -> None:
        if not source.is_file() or source.is_symlink():
            raise StageArtifactIntegrityError(
                f"cache object is not a regular file: {source}"
            )
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
        with (
            source.open("rb") as input_handle,
            destination.open("xb") as output_handle,
        ):
            while chunk := input_handle.read(self.limits.transfer_chunk_bytes):
                output_handle.write(chunk)
            output_handle.flush()
            os.fsync(output_handle.fileno())

    def _verify_existing_object(
        self,
        path: Path,
        *,
        digest: str,
        expected_size: int,
    ) -> None:
        if not path.is_file() or path.is_symlink():
            raise StageArtifactIntegrityError(
                f"immutable cache object is not a regular file: {path}"
            )
        if path.stat().st_size != expected_size:
            raise StageArtifactIntegrityError(
                f"immutable cache object has the wrong size: {digest}"
            )
        if _sha256_file(path, self.limits.transfer_chunk_bytes) != digest:
            raise StageArtifactIntegrityError(
                f"immutable cache object has the wrong digest: {digest}"
            )

    def _cleanup_package(
        self, package_id: str, *, prune_objects: bool
    ) -> StageArtifactCleanup:
        package_path = self.root / "packages" / package_id
        object_digests: list[str] = []
        package_removed = False
        destination_key = hashlib.sha256(str(package_path).encode("utf-8")).hexdigest()
        with _exclusive_file_lock(
            self._lock_path("materialize", destination_key),
            timeout_seconds=self.limits.lock_timeout_seconds,
        ):
            with self._catalog_lock():
                if package_path.exists():
                    verified = verify_stage_artifact(
                        package_path, expected_package_id=package_id
                    )
                    manifest_bytes = (
                        package_path / STAGE_ARTIFACT_MANIFEST
                    ).read_bytes()
                    object_digests.append(hashlib.sha256(manifest_bytes).hexdigest())
                    object_digests.extend(
                        str(item["sha256"])
                        for item in verified.manifest.to_document()["files"]
                    )
                    for entry in package_path.iterdir():
                        entry.unlink()
                    package_path.rmdir()
                    _fsync_directory(package_path.parent)
                    package_removed = True

                removed: list[str] = []
                partials: list[str] = []
                if prune_objects:
                    for object_digest in object_digests:
                        if self._object_is_referenced(object_digest):
                            continue
                        object_removed, removed_partials = (
                            self._remove_object_and_partial(object_digest)
                        )
                        if object_removed:
                            removed.append(object_digest)
                        partials.extend(removed_partials)
        return StageArtifactCleanup(
            cache_root=str(self.root),
            target_kind="package",
            target=package_id,
            package_removed=package_removed,
            objects_removed=tuple(removed),
            partials_removed=tuple(partials),
        )

    def _object_is_referenced(self, digest: str) -> bool:
        packages_root = self.root / "packages"
        for package_path in packages_root.iterdir():
            if package_path.name.startswith("."):
                continue
            if not package_path.is_dir() or package_path.is_symlink():
                raise StageArtifactIntegrityError(
                    f"cache package entry is not a directory: {package_path}"
                )
            manifest_path = package_path / STAGE_ARTIFACT_MANIFEST
            if not manifest_path.is_file() or manifest_path.is_symlink():
                raise StageArtifactIntegrityError(
                    f"cache package has no sealed manifest: {package_path}"
                )
            manifest_bytes = manifest_path.read_bytes()
            if hashlib.sha256(manifest_bytes).hexdigest() == digest:
                return True
            manifest = _parse_canonical_manifest(manifest_bytes)
            if any(
                str(item["sha256"]) == digest
                for item in manifest.to_document()["files"]
            ):
                return True
        return False

    def _remove_object_and_partial(self, digest: str) -> tuple[bool, tuple[str, ...]]:
        with _exclusive_file_lock(
            self._lock_path("object", digest),
            timeout_seconds=self.limits.lock_timeout_seconds,
        ):
            object_path = self.object_path(digest)
            object_removed = False
            if object_path.exists():
                if not object_path.is_file() or object_path.is_symlink():
                    raise StageArtifactIntegrityError(
                        f"cache object is not a regular file: {object_path}"
                    )
                object_path.unlink()
                object_removed = True
            part_path, state_path = self._partial_paths(digest)
            removed_partials: list[str] = []
            for path in (part_path, state_path):
                if path.exists():
                    if not path.is_file() or path.is_symlink():
                        raise StageArtifactIntegrityError(
                            f"partial cache entry is not a regular file: {path}"
                        )
                    path.unlink()
                    removed_partials.append(str(path))
            return object_removed, tuple(removed_partials)

    def _partial_paths(self, digest: str) -> tuple[Path, Path]:
        base = self.root / "partials" / "sha256" / digest[:2] / digest
        return base.with_suffix(".part"), base.with_suffix(".part.json")

    def _lock_path(self, category: str, key: str) -> Path:
        return self.root / "locks" / category / key[:2] / f"{key}.lock"

    @contextmanager
    def _catalog_lock(self) -> Iterator[None]:
        key = "0" * 64
        with _exclusive_file_lock(
            self._lock_path("catalog", key),
            timeout_seconds=self.limits.lock_timeout_seconds,
        ):
            yield


def prepare_stage_artifact(
    manifest_source: str | os.PathLike[str],
    cache_root: str | os.PathLike[str],
    *,
    destination: str | os.PathLike[str] | None = None,
    expected_package_id: str | None = None,
    limits: StageArtifactCacheLimits | None = None,
) -> StageArtifactAcquisition:
    """Prepare and verify a native stage directory before creating StageRunner."""

    return StageArtifactCache(cache_root, limits=limits).acquire(
        manifest_source,
        destination=destination,
        expected_package_id=expected_package_id,
    )


def _parse_canonical_manifest(value: bytes) -> SafeTensorsStageArtifact:
    try:
        document = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StageArtifactIntegrityError(
            "stage manifest is not valid UTF-8 JSON"
        ) from error
    try:
        manifest = parse_stage_artifact_manifest(document)
    except (TypeError, ValueError) as error:
        raise StageArtifactIntegrityError(
            f"stage manifest contract is invalid: {error}"
        ) from error
    canonical = (
        json.dumps(
            manifest.to_document(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    if value != canonical:
        raise StageArtifactIntegrityError("stage manifest encoding is not canonical")
    return manifest


def _parse_partial_state(value: bytes) -> _PartialState:
    try:
        document = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StageArtifactIntegrityError(
            "partial state is not valid UTF-8 JSON"
        ) from error
    expected_keys = {
        "schema",
        "digest",
        "expectedSizeBytes",
        "source",
        "version",
        "etag",
        "confirmedOffset",
    }
    if not isinstance(document, dict) or set(document) != expected_keys:
        raise StageArtifactIntegrityError("partial state has unknown or missing fields")
    if document["schema"] != STAGE_CACHE_PARTIAL_SCHEMA:
        raise StageArtifactIntegrityError("unsupported partial state schema")
    digest = _digest(document["digest"], "partial digest")
    expected_size = _nonnegative_integer(
        document["expectedSizeBytes"], "partial expected size"
    )
    confirmed_offset = _nonnegative_integer(
        document["confirmedOffset"], "partial confirmed offset"
    )
    if confirmed_offset > expected_size:
        raise StageArtifactIntegrityError(
            "partial confirmed offset exceeds expected size"
        )
    source = document["source"]
    if not isinstance(source, str) or not source:
        raise StageArtifactIntegrityError("partial source must be a string")
    version = document["version"]
    etag = document["etag"]
    if version is not None and (not isinstance(version, str) or not version):
        raise StageArtifactIntegrityError("partial version must be nullable text")
    if etag is not None and (not isinstance(etag, str) or not etag):
        raise StageArtifactIntegrityError("partial ETag must be nullable text")
    canonical = (
        json.dumps(
            document,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    if value != canonical:
        raise StageArtifactIntegrityError("partial state encoding is not canonical")
    return _PartialState(
        digest=digest,
        expected_size=expected_size,
        source=source,
        version=version,
        etag=etag,
        confirmed_offset=confirmed_offset,
    )


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise StageArtifactIntegrityError(f"{name} must be a non-negative integer")
    return value


def _optional_content_length(headers: Mapping[str, str]) -> int | None:
    value = headers.get("Content-Length")
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise StageArtifactIntegrityError(
            "HTTP Content-Length is not an integer"
        ) from error
    if parsed < 0:
        raise StageArtifactIntegrityError("HTTP Content-Length cannot be negative")
    return parsed


def _strong_etag(value: str | None) -> str:
    if (
        value is None
        or value.startswith("W/")
        or len(value) < 2
        or not value.startswith('"')
        or not value.endswith('"')
    ):
        raise StageArtifactIntegrityError("HTTP stage payload requires a strong ETag")
    return value


def _reject_encoded_response(headers: Mapping[str, str]) -> None:
    encoding = headers.get("Content-Encoding")
    if encoding is not None and encoding.lower() != "identity":
        raise StageArtifactIntegrityError(
            "encoded HTTP payloads cannot satisfy byte-addressed artifacts"
        )


def _validate_content_range(
    value: str | None, *, offset: int, expected_size: int
) -> None:
    match = _CONTENT_RANGE_PATTERN.fullmatch(value or "")
    if match is None:
        raise StageArtifactIntegrityError("HTTP resume has no valid Content-Range")
    start, end, total = (int(item) for item in match.groups())
    if start != offset or end != expected_size - 1 or total != expected_size:
        raise StageArtifactIntegrityError(
            "HTTP Content-Range does not match the confirmed offset"
        )


def _local_source_version(value: os.stat_result) -> str:
    return (
        f"dev={value.st_dev};ino={value.st_ino};size={value.st_size};"
        f"mtime_ns={value.st_mtime_ns}"
    )


def _sha256_file(path: Path, chunk_bytes: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    if sys.platform == "win32":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _exclusive_file_lock(path: Path, *, timeout_seconds: float) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                _try_lock_file(handle)
                break
            except OSError as error:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out waiting for cache lock: {path}"
                    ) from error
                time.sleep(0.05)
        try:
            yield
        finally:
            _unlock_file(handle)


def _try_lock_file(handle: BinaryIO) -> None:
    handle.seek(0)
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(handle: BinaryIO) -> None:
    handle.seek(0)
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_limits = StageArtifactCacheLimits()
    parser = argparse.ArgumentParser(
        description=("Acquire or selectively clean native Mycellios stage artifacts.")
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    acquire = subparsers.add_parser(
        "acquire",
        help="stream, authenticate, cache, and materialize a stage package",
    )
    acquire.add_argument("--cache-root", required=True)
    acquire.add_argument("--manifest", required=True)
    acquire.add_argument("--destination")
    acquire.add_argument("--expected-package-id")
    acquire.add_argument(
        "--max-payload-bytes",
        type=int,
        default=default_limits.max_payload_bytes,
    )
    acquire.add_argument(
        "--max-package-bytes",
        type=int,
        default=default_limits.max_package_bytes,
    )

    cleanup = subparsers.add_parser(
        "cleanup", help="remove one exact package or unreferenced digest"
    )
    cleanup.add_argument("--cache-root", required=True)
    target = cleanup.add_mutually_exclusive_group(required=True)
    target.add_argument("--package-id")
    target.add_argument("--digest")
    cleanup.add_argument("--prune-package-objects", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "acquire":
        limits = StageArtifactCacheLimits(
            max_payload_bytes=args.max_payload_bytes,
            max_package_bytes=args.max_package_bytes,
        )
        result = prepare_stage_artifact(
            args.manifest,
            args.cache_root,
            destination=args.destination,
            expected_package_id=args.expected_package_id,
            limits=limits,
        )
    else:
        cache = StageArtifactCache(args.cache_root)
        result = cache.cleanup(
            package_id=args.package_id,
            digest=args.digest,
            prune_package_objects=args.prune_package_objects,
        )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "STAGE_CACHE_PARTIAL_SCHEMA",
    "CachedStageObject",
    "StageArtifactAcquisition",
    "StageArtifactCache",
    "StageArtifactCacheError",
    "StageArtifactCacheLimits",
    "StageArtifactCleanup",
    "StageArtifactIntegrityError",
    "StageArtifactTransportError",
    "main",
    "prepare_stage_artifact",
]
