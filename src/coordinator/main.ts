import { loadCoordinatorConfig } from "../core/config.js";
import { createCoordinator } from "./server.js";

const config = loadCoordinatorConfig();
const runtime = await createCoordinator(config, { logger: true });

await runtime.app.listen({ host: config.host, port: config.port });
runtime.app.log.info(
  `GPU Distribuida coordinator listening on http://${config.host}:${config.port}`,
);

for (const signal of ["SIGINT", "SIGTERM"] as const) {
  process.once(signal, () => {
    void runtime.close().finally(() => process.exit(0));
  });
}
