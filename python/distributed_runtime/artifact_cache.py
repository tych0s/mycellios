from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import BinaryIO, Iterator

from .stage_artifact import (
    STAGE_ARTIFACT_MANIFEST,
    parse_stage_artifact_manifest,
    verify_stage_artifact,
)


ARTIFACT_CACHE_SCHEMA = "mycellios-content-addressed-cache/1"
_COPY_BUFFER_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class CachedStageArtifact:
    package_id: str
    artifact_identity: str
    directory: Path
    payload_size_bytes: int


class ResumableBlobTransfer:
    """One append-only, offset-confirmed transfer into the shared blob store."""

    def __init__(
        self,
        cache: "ContentAddressedStageCache",
        digest: str,
        expected_size: int,
    ) -> None:
        self._cache = cache
        self.digest = _digest(digest)
        self.expected_size = _positive_size(expected_size)
        self._final = cache.blob_path(self.digest)
        self._partial = cache.partial_path(self.digest)
        self._metadata = cache.partial_metadata_path(self.digest)
        self._prepare()

    @property
    def complete(self) -> bool:
        return self._final.is_file()

    @property
    def confirmed_offset(self) -> int:
        if self.complete:
            return self.expected_size
        try:
            size = self._partial.stat().st_size
        except FileNotFoundError:
            return 0
        if size > self.expected_size:
            raise ValueError("partial blob is larger than its sealed size")
        return size

    def append(
        self,
        offset: int,
        data: bytes,
        *,
        chunk_sha256: str,
    ) -> int:
        if self.complete:
            if offset == self.expected_size and not data:
                return self.expected_size
            raise FileExistsError("content-addressed blob is already complete")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("blob offset must be a non-negative integer")
        if not isinstance(data, bytes):
            raise TypeError("blob chunks must be bytes")
        if not data:
            raise ValueError("blob chunks cannot be empty")
        expected_chunk = _digest(chunk_sha256)
        if hashlib.sha256(data).hexdigest() != expected_chunk:
            raise ValueError("blob chunk digest mismatch")
        current = self.confirmed_offset
        if offset != current:
            raise ValueError(
                f"blob offset mismatch: confirmed={current}, received={offset}"
            )
        if current + len(data) > self.expected_size:
            raise ValueError("blob chunk exceeds the sealed size")
        self._partial.parent.mkdir(parents=True, exist_ok=True)
        with self._partial.open("ab", buffering=0) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return self.confirmed_offset

    def finalize(self) -> Path:
        if self.complete:
            self._cache.verify_blob(
                self.digest,
                expected_size=self.expected_size,
            )
            return self._final
        if self.confirmed_offset != self.expected_size:
            raise ValueError(
                "blob is incomplete: "
                f"confirmed={self.confirmed_offset}, expected={self.expected_size}"
            )
        if _sha256_file(self._partial) != self.digest:
            raise ValueError("completed blob digest mismatch")
        self._final.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._partial.replace(self._final)
        except FileExistsError:
            self._cache.verify_blob(
                self.digest,
                expected_size=self.expected_size,
            )
            self._partial.unlink(missing_ok=True)
        self._metadata.unlink(missing_ok=True)
        self._cache.verify_blob(self.digest, expected_size=self.expected_size)
        return self._final

    def _prepare(self) -> None:
        if self._final.exists():
            self._cache.verify_blob(
                self.digest,
                expected_size=self.expected_size,
            )
            self._partial.unlink(missing_ok=True)
            self._metadata.unlink(missing_ok=True)
            return
        self._partial.parent.mkdir(parents=True, exist_ok=True)
        expected_metadata = {
            "schema": ARTIFACT_CACHE_SCHEMA,
            "sha256": self.digest,
            "sizeBytes": self.expected_size,
        }
        if self._metadata.exists():
            actual = _read_json_object(self._metadata, "blob transfer metadata")
            if actual != expected_metadata:
                raise ValueError("partial blob metadata does not match the request")
        else:
            if self._partial.exists():
                raise ValueError("partial blob exists without sealed metadata")
            _write_json_atomic(self._metadata, expected_metadata)
        if self._partial.exists() and self._partial.stat().st_size > self.expected_size:
            raise ValueError("partial blob is larger than its sealed size")


class ContentAddressedStageCache:
    """Verified local store shared by model stages and peer transfers.

    Files are keyed by SHA-256, so equal weights are stored once. A package is
    published only after all files and the canonical manifest verify. Partial
    transfers remain private and can resume after a process or network restart.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).resolve()
        self._blobs = self.root / "blobs" / "sha256"
        self._partials = self.root / "partial" / "sha256"
        self._packages = self.root / "packages" / "sha256"
        for directory in (self._blobs, self._partials, self._packages):
            directory.mkdir(parents=True, exist_ok=True)

    def blob_path(self, digest: str) -> Path:
        value = _digest(digest)
        return self._blobs / value[:2] / value

    def partial_path(self, digest: str) -> Path:
        value = _digest(digest)
        return self._partials / value[:2] / f"{value}.part"

    def partial_metadata_path(self, digest: str) -> Path:
        value = _digest(digest)
        return self._partials / value[:2] / f"{value}.json"

    def package_path(self, package_id: str) -> Path:
        value = _digest(package_id)
        return self._packages / value[:2] / value

    def begin_blob(
        self,
        digest: str,
        expected_size: int,
    ) -> ResumableBlobTransfer:
        return ResumableBlobTransfer(self, digest, expected_size)

    def verify_blob(
        self,
        digest: str,
        *,
        expected_size: int | None = None,
    ) -> Path:
        value = _digest(digest)
        path = self.blob_path(value)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"content-addressed blob is missing: {value}")
        if expected_size is not None and path.stat().st_size != _positive_size(
            expected_size
        ):
            raise ValueError("content-addressed blob size mismatch")
        if _sha256_file(path) != value:
            raise ValueError("content-addressed blob digest mismatch")
        return path

    def import_stage(
        self,
        directory: str | os.PathLike[str],
    ) -> CachedStageArtifact:
        verified = verify_stage_artifact(directory)
        document = verified.manifest.to_document()
        root = verified.root
        for item in document["files"]:
            source = root / item["path"]
            self._import_blob(
                source,
                digest=item["sha256"],
                expected_size=item["sizeBytes"],
            )
        manifest_bytes = (root / STAGE_ARTIFACT_MANIFEST).read_bytes()
        return self.commit_manifest(manifest_bytes)

    def commit_manifest(self, manifest_bytes: bytes) -> CachedStageArtifact:
        document = _canonical_manifest_document(manifest_bytes)
        manifest = parse_stage_artifact_manifest(document)
        canonical = _canonical_manifest_bytes(document)
        if manifest_bytes != canonical:
            raise ValueError("stage artifact manifest encoding is not canonical")
        payload_size = 0
        for item in document["files"]:
            self.verify_blob(item["sha256"], expected_size=item["sizeBytes"])
            payload_size += item["sizeBytes"]
        manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
        self._import_bytes(manifest_bytes, manifest_digest)

        target = self.package_path(manifest.package_id)
        if target.exists():
            verified = verify_stage_artifact(
                target,
                expected_package_id=manifest.package_id,
            )
            return CachedStageArtifact(
                package_id=verified.manifest.package_id,
                artifact_identity=f"sha256:{verified.manifest.package_id}",
                directory=target,
                payload_size_bytes=payload_size,
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.publish-", dir=target.parent)
        )
        try:
            self._link_or_copy(
                self.verify_blob(
                    manifest_digest,
                    expected_size=len(manifest_bytes),
                ),
                temporary / STAGE_ARTIFACT_MANIFEST,
            )
            for item in document["files"]:
                self._link_or_copy(
                    self.verify_blob(
                        item["sha256"],
                        expected_size=item["sizeBytes"],
                    ),
                    temporary / item["path"],
                )
            verify_stage_artifact(
                temporary,
                expected_package_id=manifest.package_id,
            )
            try:
                temporary.replace(target)
            except FileExistsError:
                verify_stage_artifact(
                    target,
                    expected_package_id=manifest.package_id,
                )
                shutil.rmtree(temporary, ignore_errors=True)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return CachedStageArtifact(
            package_id=manifest.package_id,
            artifact_identity=f"sha256:{manifest.package_id}",
            directory=target,
            payload_size_bytes=payload_size,
        )

    def materialize_stage(
        self,
        package_id: str,
        destination: str | os.PathLike[str],
    ) -> Path:
        value = _digest(package_id)
        source = self.package_path(value)
        verify_stage_artifact(source, expected_package_id=value)
        target = Path(destination).resolve()
        if target.exists():
            raise FileExistsError(f"stage destination already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.materialize-", dir=target.parent)
        )
        try:
            for item in source.iterdir():
                if not item.is_file() or item.is_symlink():
                    raise ValueError("cached package contains an invalid entry")
                self._link_or_copy(item, temporary / item.name)
            verify_stage_artifact(temporary, expected_package_id=value)
            temporary.replace(target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return target

    def cached_packages(self) -> tuple[CachedStageArtifact, ...]:
        packages: list[CachedStageArtifact] = []
        for prefix in sorted(self._packages.iterdir()):
            if not prefix.is_dir() or prefix.is_symlink():
                continue
            for path in sorted(prefix.iterdir()):
                if not path.is_dir() or path.is_symlink():
                    continue
                verified = verify_stage_artifact(path, expected_package_id=path.name)
                payload_size = sum(
                    item["sizeBytes"]
                    for item in verified.manifest.to_document()["files"]
                )
                packages.append(
                    CachedStageArtifact(
                        package_id=verified.manifest.package_id,
                        artifact_identity=f"sha256:{verified.manifest.package_id}",
                        directory=path,
                        payload_size_bytes=payload_size,
                    )
                )
        return tuple(packages)

    def _import_blob(
        self,
        source: Path,
        *,
        digest: str,
        expected_size: int,
    ) -> Path:
        transfer = self.begin_blob(digest, expected_size)
        if transfer.complete:
            return transfer.finalize()
        with source.open("rb") as stream:
            stream.seek(transfer.confirmed_offset)
            for data in _chunks(stream):
                transfer.append(
                    transfer.confirmed_offset,
                    data,
                    chunk_sha256=hashlib.sha256(data).hexdigest(),
                )
        return transfer.finalize()

    def _import_bytes(self, value: bytes, digest: str) -> Path:
        transfer = self.begin_blob(digest, len(value))
        if not transfer.complete and transfer.confirmed_offset < len(value):
            remainder = value[transfer.confirmed_offset :]
            transfer.append(
                transfer.confirmed_offset,
                remainder,
                chunk_sha256=hashlib.sha256(remainder).hexdigest(),
            )
        return transfer.finalize()

    @staticmethod
    def _link_or_copy(source: Path, destination: Path) -> None:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def _canonical_manifest_document(value: bytes) -> dict[str, object]:
    try:
        document = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("stage artifact manifest is not valid JSON") from error
    if not isinstance(document, dict):
        raise ValueError("stage artifact manifest is not an object")
    return document


def _canonical_manifest_bytes(document: dict[str, object]) -> bytes:
    return (
        json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _chunks(stream: BinaryIO) -> Iterator[bytes]:
    while data := stream.read(_COPY_BUFFER_BYTES):
        yield data


def _digest(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("SHA-256 digest must be text")
    normalized = value.removeprefix("sha256:").lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("SHA-256 digest must contain exactly 64 hexadecimal digits")
    return normalized


def _positive_size(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("blob size must be a positive integer")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in _chunks(stream):
            digest.update(data)
    return digest.hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    return value


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


__all__ = [
    "ARTIFACT_CACHE_SCHEMA",
    "CachedStageArtifact",
    "ContentAddressedStageCache",
    "ResumableBlobTransfer",
]
