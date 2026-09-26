import { AppWindow, ChevronRight, Download, MonitorUp } from "lucide-react";
import { LandingHeader } from "./RebrandLanding";
import "./earn-studio.css";

const contributionOptions = [
  {
    id: "browser",
    eyebrow: "IN THIS BROWSER",
    title: "Start a browser worker",
    copy: "Use WebGPU when available, with a CPU fallback. This tab stays open while your device contributes.",
    detail: "No installation · Pause at any time",
    action: "Open browser worker",
    href: "/browser/?autostart=1",
    icon: AppWindow,
  },
  {
    id: "native",
    eyebrow: "ON MY MACHINE",
    title: "Native worker packages",
    copy: "Run verified workloads with compatible GPU acceleration when a native package is published for your system.",
    detail: "Windows · macOS · Linux",
    action: "Check package availability",
    href: "/downloads",
    icon: MonitorUp,
  },
] as const;

export function EarnStudio() {
  return (
    <main className="rb-page earn-page">
      <LandingHeader active="earn" />

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
