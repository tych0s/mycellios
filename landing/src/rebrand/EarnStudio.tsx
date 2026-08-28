import { AppWindow, ChevronRight, Download, Menu, MonitorUp, X } from "lucide-react";
import { useState } from "react";
import { SporeMenu, SporeMobileLinks } from "./SporeMenu";
import "./create-studio.css";
import "./earn-studio.css";

const contributionOptions = [
  {
    id: "browser",
    eyebrow: "IN THIS BROWSER",
    title: "Start a browser worker",
    copy: "Use WebGPU when available, with a CPU fallback. This tab stays open while your device contributes.",
    detail: "No installation · Pause at any time",
    action: "Open browser worker",
    href: "/mobile/?autostart=1",
    icon: AppWindow,
  },
  {
    id: "native",
    eyebrow: "ON MY MACHINE",
    title: "Install the native worker",
    copy: "Run verified workloads through the Mycellios desktop runtime, including compatible GPU acceleration.",
    detail: "Windows · macOS · Linux",
    action: "Choose an installer",
    href: "/downloads",
    icon: MonitorUp,
  },
] as const;

export function EarnStudio() {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <main className="earn-page">
      <header className="create-header earn-header">
        <a className="create-brand" href="/" aria-label="Mycellios home"><img src="/assets/logos/logo.png" alt="" />mycellios</a>
        <nav aria-label="Main navigation">
          <a href="/network?view=inference">Chat</a><a href="/create">Create</a><a className="active" href="/earn" aria-current="page">Earn</a><SporeMenu /><a href="/#how-it-works">Network</a><a href="/blog">Blog</a>
        </nav>
        <a className="create-login" href="/network?view=overview">Login</a>
        <button className="create-menu-button" type="button" aria-label={menuOpen ? "Close menu" : "Open menu"} aria-expanded={menuOpen} onClick={() => setMenuOpen((open) => !open)}>{menuOpen ? <X /> : <Menu />}</button>
        {menuOpen && <div className="create-mobile-nav"><a href="/network?view=inference">Chat</a><a href="/create">Create</a><a className="active" href="/earn">Earn</a><SporeMobileLinks /><a href="/#how-it-works">Network</a><a href="/blog">Blog</a><a href="/network?view=overview">Login</a></div>}
      </header>

      <section className="earn-intro">
        <span>CONTRIBUTE CAPACITY</span>
        <h1>Put your machine to work.</h1>
        <p>Choose how this device joins the Mycellios test network. You stay in control of what it shares and when it stops.</p>
      </section>

      <section className="earn-options" aria-label="Contribution options">
        {contributionOptions.map((option) => {
          const Icon = option.icon;
          return <article className={`earn-option ${option.id}`} key={option.id}>
            <div className="earn-option-icon"><Icon /></div>
            <div className="earn-option-copy"><span>{option.eyebrow}</span><h2>{option.title}</h2><p>{option.copy}</p><small>{option.detail}</small></div>
            <a href={option.href}>{option.id === "native" && <Download />}<span>{option.action}</span><ChevronRight /></a>
          </article>;
        })}
      </section>

      <aside className="earn-notice"><i aria-hidden="true" /><div><strong>Test network · rewards are not live yet</strong><span>Contribution is measured for reliability and useful work. No USDC or token payment is currently promised.</span></div></aside>
    </main>
  );
}
