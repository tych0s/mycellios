from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Literal


STAGE_FAILURE_EVIDENCE_SCHEMA = "gdlp-stage-failure-evidence/1"
_EXECUTOR_ID = re.compile(r"^[0-9a-f]{32}$")
_ROUTE_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_ERROR_TYPE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,127}$")
StageFailureRole = Literal["head", "middle", "tail"]


@dataclass(frozen=True, slots=True)
class StageFailureEvidence:
    generation: int
    route_id: str
    stage_role: StageFailureRole
    stage_index: int
    executor_id: str
    layer_start: int
    layer_end: int
    failure_class: Literal["health-failed"]
    error_type: str
    message: str

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("generation", self.generation, 0),
            ("stage_index", self.stage_index, 1),
            ("layer_start", self.layer_start, 0),
            ("layer_end", self.layer_end, 1),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < minimum
                or value > (1 << 64) - 1
            ):
                raise ValueError(f"{name} is invalid")
        if self.layer_end <= self.layer_start:
            raise ValueError("stage failure layer range is invalid")
        if not _ROUTE_ID.fullmatch(self.route_id):
            raise ValueError("stage failure route id is invalid")
        if self.stage_role not in {"head", "middle", "tail"}:
            raise ValueError("stage failure role is invalid")
        if not _EXECUTOR_ID.fullmatch(self.executor_id):
            raise ValueError("stage failure executor id is invalid")
        if self.failure_class != "health-failed":
            raise ValueError("stage failure class is invalid")
        if not _ERROR_TYPE.fullmatch(self.error_type):
            raise ValueError("stage failure error type is invalid")
        if not isinstance(self.message, str) or not self.message or len(self.message) > 1024:
            raise ValueError("stage failure message is invalid")

    def to_document(self) -> dict[str, object]:
        value = asdict(self)
        return {
            "schema": STAGE_FAILURE_EVIDENCE_SCHEMA,
            "generation": value["generation"],
            "routeId": value["route_id"],
            "stageRole": value["stage_role"],
            "stageIndex": value["stage_index"],
            "executorId": value["executor_id"],
            "layerStart": value["layer_start"],
            "layerEnd": value["layer_end"],
            "failureClass": value["failure_class"],
            "errorType": value["error_type"],
            "message": value["message"],
        }

    def to_payload(self) -> bytes:
        return _canonical_json(self.to_document())

    @classmethod
    def parse_payload(cls, payload: bytes | bytearray) -> "StageFailureEvidence":
        raw = bytes(payload)
        try:
            document = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("stage failure evidence is not canonical JSON") from error
        if not isinstance(document, dict) or set(document) != {
            "schema", "generation", "routeId", "stageRole", "stageIndex",
            "executorId", "layerStart", "layerEnd", "failureClass",
            "errorType", "message",
        }:
            raise ValueError("stage failure evidence has unknown or missing fields")
        if document["schema"] != STAGE_FAILURE_EVIDENCE_SCHEMA:
            raise ValueError("stage failure evidence schema is unsupported")
        evidence = cls(
            generation=document["generation"], route_id=document["routeId"],
            stage_role=document["stageRole"], stage_index=document["stageIndex"],
            executor_id=document["executorId"], layer_start=document["layerStart"],
            layer_end=document["layerEnd"], failure_class=document["failureClass"],
            error_type=document["errorType"], message=document["message"],
        )
        if raw != evidence.to_payload():
            raise ValueError("stage failure evidence encoding is not canonical")
        return evidence


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


__all__ = ["STAGE_FAILURE_EVIDENCE_SCHEMA", "StageFailureEvidence"]
