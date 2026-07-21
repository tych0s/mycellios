import { createRoot } from "react-dom/client";
import { Panel } from "../../landing/src/Panel";
import "./styles.css";

const root = document.getElementById("root");
if (!root) throw new Error("Missing mobile root element");

createRoot(root).render(<Panel mobileEntry />);

if ("serviceWorker" in navigator && window.isSecureContext) {
  window.addEventListener("load", () => {
    void navigator.serviceWorker.register("./sw.js", { scope: "./" });
  });
}
