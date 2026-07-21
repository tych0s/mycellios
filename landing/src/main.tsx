import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Landing } from "./Landing";
import { Panel } from "./Panel";
import "./styles.css";

const root = document.getElementById("root");
if (!root) throw new Error("Missing landing root element");

const panelRoutes = new Set(["/network", "/admin", "/join", "/downloads"]);

createRoot(root).render(
  <StrictMode>
    {panelRoutes.has(window.location.pathname) ? <Panel /> : <Landing />}
  </StrictMode>,
);
