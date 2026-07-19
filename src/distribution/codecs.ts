import type { ActivationCodec, ActivationCodecId } from "./types.js";

export const ACTIVATION_CODECS: Readonly<Record<ActivationCodecId, ActivationCodec>> = {
  fp16: {
    id: "fp16",
    bytesPerElement: 2,
    encodeGbps: Number.POSITIVE_INFINITY,
    decodeGbps: Number.POSITIVE_INFINITY,
    fixedEncodeMs: 0,
    fixedDecodeMs: 0,
    estimatedQualityLoss: 0,
  },
  int8: {
    id: "int8",
    bytesPerElement: 1,
    encodeGbps: 12,
    decodeGbps: 18,
    fixedEncodeMs: 0.015,
    fixedDecodeMs: 0.01,
    estimatedQualityLoss: 0.001,
  },
  "int8-grouped": {
    id: "int8-grouped",
    // One FP32 scale per 64 values adds approximately 0.0625 bytes/element.
    bytesPerElement: 1.0625,
    encodeGbps: 10,
    decodeGbps: 14,
    fixedEncodeMs: 0.02,
    fixedDecodeMs: 0.015,
    // Until a multi-model calibration exists this stays as conservative as INT8.
    estimatedQualityLoss: 0.001,
  },
  "int8-hadamard": {
    id: "int8-hadamard",
    bytesPerElement: 1.0625,
    encodeGbps: 7,
    decodeGbps: 9,
    fixedEncodeMs: 0.03,
    fixedDecodeMs: 0.025,
    estimatedQualityLoss: 0.001,
  },
  q4: {
    id: "q4",
    bytesPerElement: 0.5,
    encodeGbps: 7,
    decodeGbps: 11,
    fixedEncodeMs: 0.025,
    fixedDecodeMs: 0.02,
    estimatedQualityLoss: 0.02,
  },
};

export function activationCodec(id: ActivationCodecId): ActivationCodec {
  return ACTIVATION_CODECS[id];
}
