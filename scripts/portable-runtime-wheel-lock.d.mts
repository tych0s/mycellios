export const PORTABLE_RUNTIME_WHEEL_LOCK_SCHEMA:
  "mycellios-python-wheel-lock/1";

export interface PortableRuntimeWheelLockArtifact {
  readonly name: string;
  readonly version: string;
  readonly url: string;
  readonly sha256: string;
  readonly filename: string;
}

export interface PortableRuntimeWheelLock {
  readonly schema: typeof PORTABLE_RUNTIME_WHEEL_LOCK_SCHEMA;
  readonly platform: string;
  readonly arch: string;
  readonly sha256: string;
  readonly path: string;
  readonly artifacts: readonly PortableRuntimeWheelLockArtifact[];
}

export interface PortableRuntimeWheelLockSpec {
  readonly supported: boolean;
  readonly platform: string;
  readonly arch: string;
  readonly torchVersion?: string;
  readonly packageVersions?: Readonly<Record<string, string>>;
  readonly wheelLock?: {
    readonly schema: typeof PORTABLE_RUNTIME_WHEEL_LOCK_SCHEMA;
    readonly path: string;
    readonly sha256: string;
  };
}

export function readPortableRuntimeWheelLock(
  workspaceRoot: string,
  spec: PortableRuntimeWheelLockSpec,
): PortableRuntimeWheelLock;

export function parsePortableRuntimeWheelLock(
  source: string,
  spec: PortableRuntimeWheelLockSpec,
): PortableRuntimeWheelLock;

export function portableRuntimeOfflineRequirements(
  lock: PortableRuntimeWheelLock,
): string;

export function verifyPortableRuntimeWheelhouse(
  directory: string,
  lock: PortableRuntimeWheelLock,
): Promise<true>;
