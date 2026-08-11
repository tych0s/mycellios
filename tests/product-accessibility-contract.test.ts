import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const panel = readFileSync(new URL("../landing/src/Panel.tsx", import.meta.url), "utf8");
const panelCss = readFileSync(new URL("../landing/src/panel.css", import.meta.url), "utf8");
const brandCss = readFileSync(new URL("../landing/src/rebrand/rebrand.css", import.meta.url), "utf8");

describe("product accessibility contract", () => {
  it("keeps the application shell and page hierarchy semantic", () => {
    expect(panel).toContain('<aside id="panel-navigation"');
    expect(panel).toContain('aria-label="Primary navigation"');
    expect(panel).toContain('className="panel-skip-link" href="#panel-main-content"');
    expect(panel).toContain('<main id="panel-main-content" ref={mainContentRef} className="panel-content" tabIndex={-1}>');
    expect(panel).toContain('mainContentRef.current?.focus()');
    expect(panel).toContain('className="panel-error" role="alert" aria-live="assertive"');
    expect(panel).toContain('className={`product-state-banner tone-${presentation.tone} state-${presentation.id}`} role="status"');
    expect(panel).toContain("<h1>{title}</h1>");
  });

  it("makes web account and detail surfaces labelled modals with bounded keyboard focus", () => {
    expect(panel).toContain('className="account-modal" role="dialog" aria-modal="true" aria-labelledby={titleId}');
    expect(panel).toContain('className="model-detail-drawer" role="dialog" aria-modal="true" aria-labelledby={titleId}');
    expect(panel).toContain('firstFocusable(dialog)?.focus() ?? dialog.focus()');
    expect(panel).toContain('trapDialogTab(event, dialog)');
    expect(panel).toContain('role="alert"');
  });

  it("preserves visible focus and reduced-motion alternatives", () => {
    expect(panelCss).toMatch(/\.public-panel button:focus-visible/);
    expect(panelCss).toMatch(/\.public-panel .*summary:focus-visible/);
    expect(panelCss).toContain(".public-panel .panel-skip-link:focus");
    expect(panelCss).toMatch(/@media \(pointer: coarse\)/);
    expect(panelCss).toMatch(/@media \(prefers-reduced-motion: reduce\)/);
    expect(brandCss).toMatch(/\.rb-page :focus-visible/);
    expect(brandCss).toMatch(/@media\(prefers-reduced-motion:reduce\)/);
  });

  it("announces canonical operator evidence without flattening heading hierarchy", () => {
    expect(panel).toContain('aria-labelledby="operator-evidence-title" aria-busy={busy}');
    expect(panel).toContain('<h2 id="operator-evidence-title">Operator evidence</h2>');
    expect(panel).toContain('role="status" aria-live="polite"');
    expect(panel).toContain('<h3>{title}</h3>');
  });
});
