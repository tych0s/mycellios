const DIALOG_FOCUS_SELECTOR = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

export function dialogFocusables(dialog: HTMLElement): HTMLElement[] {
  return [...dialog.querySelectorAll<HTMLElement>(DIALOG_FOCUS_SELECTOR)]
    .filter((element) => !element.hasAttribute("hidden") && element.getAttribute("aria-hidden") !== "true");
}

export function firstFocusable(dialog: HTMLElement): HTMLElement | null {
  return dialogFocusables(dialog)[0] ?? null;
}

export function trapDialogTab(event: KeyboardEvent, dialog: HTMLElement): void {
  if (event.key !== "Tab") return;
  const controls = dialogFocusables(dialog);
  const first = controls[0], last = controls.at(-1);
  if (!first || !last) { event.preventDefault(); dialog.focus(); return; }
  if (event.shiftKey && (document.activeElement === first || !dialog.contains(document.activeElement))) {
    event.preventDefault(); last.focus();
  } else if (!event.shiftKey && (document.activeElement === last || !dialog.contains(document.activeElement))) {
    event.preventDefault(); first.focus();
  }
}

export function formatCompactNumber(value: number): string {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: value < 10 ? 1 : 0 }).format(value);
}
export function formatCompactTokens(value: number): string {
  if (value >= 1_000_000) return `${new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 }).format(value / 1_000_000)}M`;
  if (value >= 1_000) return `${new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 }).format(value / 1_000)}K`;
  return new Intl.NumberFormat("en-US").format(value);
}
export function formatPower(watts: number): string { return watts >= 1_000 ? `${(watts / 1_000).toFixed(1)} kW` : `${Math.round(watts)} W`; }
export function shortId(value: string): string { return value.length <= 10 ? value : `${value.slice(0, 6)}…${value.slice(-4)}`; }
export function shortFingerprint(value: string): string {
  const digest = value.replace(/^sha256:/, "");
  return digest.length <= 20 ? digest : `${digest.slice(0, 10)}…${digest.slice(-8)}`;
}
export function formatMemory(value: number): string { return value >= 1_024 ? `${(value / 1_024).toFixed(value >= 10_240 ? 0 : 1)} GB` : `${Math.round(value)} MB`; }
export function relativeTime(value: string, now = Date.now()): string {
  const seconds = Math.max(0, Math.floor((now - new Date(value).getTime()) / 1_000));
  if (seconds < 5) return "now"; if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60); return minutes < 60 ? `${minutes}m ago` : `${Math.floor(minutes / 60)}h ago`;
}
export function relativeTimeEs(value: string, now = Date.now()): string {
  const seconds = Math.max(0, Math.floor((now - new Date(value).getTime()) / 1_000));
  if (seconds < 5) return "ahora"; if (seconds < 60) return `hace ${seconds} s`;
  const minutes = Math.floor(seconds / 60); return minutes < 60 ? `hace ${minutes} min` : `hace ${Math.floor(minutes / 60)} h`;
}
