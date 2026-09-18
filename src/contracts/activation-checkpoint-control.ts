/** Workspace-local Python/TypeScript IPC; never advertised as a LAN endpoint. */
export const CHECKPOINT_CONTROL_SCHEMA = "mycellios-checkpoint-control/1";
export const CHECKPOINT_CONTROL_ENDPOINT_NAME = "activation-checkpoint.endpoint.json";
export const CHECKPOINT_CONTROL_SOCKET_NAME = "activation-checkpoint.sock";
export const CHECKPOINT_CONTROL_TOKEN_ENV = "MYCELLIOS_CHECKPOINT_CONTROL_TOKEN";
export const MAX_CHECKPOINT_CONTROL_HEADER_BYTES = 4_096;

export interface CheckpointControlEndpoint {
  schema: typeof CHECKPOINT_CONTROL_SCHEMA;
  host: "127.0.0.1";
  port: number;
  /** HMAC-SHA256(hex-decoded launch token, UTF8 schema + LF + host + LF + port). */
  signature: string;
}
