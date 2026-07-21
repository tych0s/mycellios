import { loadCoordinatorConfig } from "../core/config.js";
import { AutomaticModelActivationManager } from "./model-activation-manager.js";
import { createCoordinator } from "./server.js";

const config = loadCoordinatorConfig();
const activationConfigPath = process.env.MYCELLIOS_AUTO_DISTRIBUTE_CONFIG?.trim();
const activationManager = activationConfigPath
  ? await AutomaticModelActivationManager.fromFile(activationConfigPath)
  : undefined;
const runtime = await createCoordinator(config, {
  logger: true,
  ...(activationManager ? { activationManager } : {}),
});

await runtime.app.listen({ host: config.host, port: config.port });
runtime.app.log.info(
  `mycellios coordinator listening on http://${config.host}:${config.port}`,
);
if (activationManager) {
  runtime.app.log.info(`automatic model activation enabled with ${activationConfigPath}`);
}

for (const signal of ["SIGINT", "SIGTERM"] as const) {
  process.once(signal, () => {
    void runtime.close().finally(() => process.exit(0));
  });
}
