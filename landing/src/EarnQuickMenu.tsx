import { Check, ChevronDown, ChevronRight, Copy, Download, ExternalLink, GitFork, Network, Share2, Smartphone } from "lucide-react";
import { useEffect, useId, useRef, useState } from "react";

export function EarnQuickMenu({
  earnUrl,
  browserUrl,
  downloadsUrl,
  external,
  onOpenEarn,
}: {
  earnUrl: string;
  browserUrl: string;
  downloadsUrl: string;
  external: boolean;
  onOpenEarn: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [copyState, setCopyState] = useState<"idle" | "copied" | "error">("idle");
  const menuId = useId();
  const rootRef = useRef<HTMLDivElement>(null);
  const copyResetRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const externalProps = external ? { target: "_blank", rel: "noreferrer" } as const : {};

  useEffect(() => {
    if (!open) return;
    const closeOnOutsidePress = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", closeOnOutsidePress);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnOutsidePress);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [open]);

  useEffect(() => () => {
    if (copyResetRef.current) clearTimeout(copyResetRef.current);
  }, []);

  async function copyEarnLink() {
    try {
      await navigator.clipboard.writeText(earnUrl);
      setCopyState("copied");
    } catch {
      setCopyState("error");
    }
    if (copyResetRef.current) clearTimeout(copyResetRef.current);
    copyResetRef.current = setTimeout(() => setCopyState("idle"), 2200);
  }

  return (
    <div className={`panel-join-menu ${open ? "open" : ""}`} ref={rootRef}>
      <button
        type="button"
        className="panel-join-trigger"
        aria-label="Earn"
        aria-haspopup="dialog"
        aria-controls={menuId}
        aria-expanded={open}
        onClick={() => setOpen((current) => !current)}
      >
        <Share2 size={15} />
        <span>Earn</span>
        <ChevronDown className="panel-join-chevron" size={13} />
      </button>
      {open && (
        <section className="panel-join-popover" id={menuId} role="dialog" aria-label="Earn by contributing">
          <div className="panel-join-popover-intro">
            <span>CONTRIBUTE CAPACITY</span>
            <strong>Earn</strong>
            <p>Choose how to contribute compute. Rewards remain unavailable until verified work and payouts are activated.</p>
          </div>
          <div className="panel-join-invite">
            <div className="panel-join-invite-head">
              <span>PUBLIC CONTRIBUTION LINK</span>
              <button type="button" className={copyState} onClick={() => void copyEarnLink()} aria-live="polite">
                {copyState === "copied" ? <Check size={14} /> : <Copy size={14} />}
                {copyState === "copied" ? "Copied" : copyState === "error" ? "Try again" : "Copy"}
              </button>
            </div>
            <code><span>$</span>{earnUrl}</code>
            <small>Anyone with this link can contribute in a browser or check native package availability.</small>
          </div>
          <div className="panel-join-quick-actions">
            <button type="button" onClick={() => { setOpen(false); onOpenEarn(); }}>
              <Network size={16} />
              <span><strong>Open Earn</strong><small>See every contribution option</small></span>
              <ChevronRight size={15} />
            </button>
            <a href={browserUrl} {...externalProps}>
              <Smartphone size={16} />
              <span><strong>Browser node</strong><small>Connect without installing</small></span>
              <ExternalLink size={14} />
            </a>
            <a href={downloadsUrl} {...externalProps}>
              <Download size={16} />
              <span><strong>Native packages</strong><small>Check current availability</small></span>
              <ChevronRight size={15} />
            </a>
          </div>
          <a className="panel-join-guide" href="https://github.com/tych0s/mycellios" target="_blank" rel="noreferrer">
            <GitFork size={14} /> Setup &amp; contribute on GitHub <ExternalLink size={12} />
          </a>
        </section>
      )}
    </div>
  );
}
