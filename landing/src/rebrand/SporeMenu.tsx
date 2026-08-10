import { ChevronDown } from "lucide-react";
import "./spore-menu.css";

export function SporeMenu({ active = false }: { active?: boolean }) {
  return <details className={`spore-menu${active ? " active" : ""}`}>
    <summary aria-label="Open SPORE menu"><span>$ SPORE</span><ChevronDown /></summary>
    <div className="spore-menu-popover">
      <a href="/spore">Staking</a>
      <a href="/spore/treasury">Treasury</a>
      <a href="/spore/data">Data</a>
    </div>
  </details>;
}

export function SporeMobileLinks({ active = false }: { active?: boolean }) {
  return <div className={`spore-mobile-links${active ? " active" : ""}`}>
    <strong>$ SPORE</strong>
    <a href="/spore">Staking</a><a href="/spore/treasury">Treasury</a><a href="/spore/data">Data</a>
  </div>;
}
