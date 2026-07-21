import { Panel } from "../../../landing/src/Panel";

/**
 * Electron and the public site render the same dashboard source. The preload
 * bridge enables native-only pages and window controls when it is available.
 */
export function App() {
  return <Panel desktopBridge={window.mycellios} />;
}
