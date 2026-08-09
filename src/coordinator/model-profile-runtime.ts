import { existsSync } from "node:fs";
import { posix, win32 } from "node:path";
import type { AutoDistributionConfig } from "../distribution/auto-distribute.js";
import { parseAutoDistributionConfig } from "../distribution/auto-distribute.js";

export interface CoordinatorModelProfileRuntimeOptions {
  runtimeRoot: string;
  environment?: NodeJS.ProcessEnv;
  platform?: NodeJS.Platform;
  exists?: (path: string) => boolean;
}

/**
 * The activation JSON is durable state and can outlive the machine that wrote
 * it. Never reuse an absolute Python or source path from that old host.
 */
export function configureCoordinatorModelProfileRuntime(
  base: AutoDistributionConfig,
  options: CoordinatorModelProfileRuntimeOptions,
): AutoDistributionConfig {
  const environment = options.environment ?? process.env;
  const exists = options.exists ?? existsSync;
  const explicit = environment.MYCELLIOS_MODEL_PROFILE_PYTHON?.trim();
  if (explicit && !exists(explicit)) {
    throw new Error(`coordinator_model_profile_python_not_found:${explicit}`);
  }
  const platform = options.platform ?? process.platform;
  const paths = platform === "win32" ? win32 : posix;
  const localRuntimeRoot = paths.resolve(
    options.runtimeRoot,
    "runtime",
    "distribution-venv",
  );
  const candidates = explicit
    ? [explicit]
    : platform === "win32"
      ? [
          paths.join(localRuntimeRoot, "python.exe"),
          paths.join(localRuntimeRoot, "Scripts", "python.exe"),
        ]
      : [
          "/usr/local/bin/python",
          "/opt/python/bin/python",
          paths.join(localRuntimeRoot, "bin", "python"),
          paths.join(localRuntimeRoot, "bin", "python3"),
        ];
  const pythonExecutable = candidates.find(exists);
  if (!pythonExecutable) {
    throw new Error("coordinator_model_profile_runtime_is_not_provisioned");
  }
  return parseAutoDistributionConfig({
    ...structuredClone(base),
    runtime: {
      ...base.runtime,
      pythonExecutable,
      pythonPath: paths.resolve(options.runtimeRoot, "python"),
    },
  });
}
