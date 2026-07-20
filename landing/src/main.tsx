import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Landing } from "./Landing";
import "./styles.css";

const root = document.getElementById("root");
if (!root) throw new Error("Missing landing root element");

createRoot(root).render(
  <StrictMode>
    <Landing />
  </StrictMode>,
);
