import { Activity, Database, Leaf, LockKeyhole, Network, ShieldCheck, Sprout } from "lucide-react";
import { LandingHeader } from "./RebrandLanding";
import "./create-studio.css";
import "./spore-studio.css";

const steps = [
  ["01", "Commit", "Choose an amount only after the SPORE ledger and custody contracts are live."],
  ["02", "Mature", "A future activation window can separate new deposits from established participation."],
  ["03", "Verify work", "Rewards must come from paid, verifiable compute—not raw usage or circular activity."],
  ["04", "Claim or compound", "Claiming and compounding remain disabled until custody, accounting and audits are complete."],
] as const;

function PreviewNotice() {
  return <aside className="spore-notice"><Sprout /><div><strong>SPORE is a product preview</strong><span>No token, staking contract, treasury, price feed or reward program is live.</span></div></aside>;
}

function Staking() {
  return <><section className="spore-hero"><div><span>NETWORK PARTICIPATION</span><h1>Grow with <em>$SPORE</em></h1><p>A proposed self-custody participation layer for useful, verified compute. Designed to reward real work—not speculative activity.</p></div><div className="spore-mark" aria-hidden="true"><i /><i /><i /><Leaf /></div></section><PreviewNotice />
    <section className="spore-gate"><LockKeyhole /><h2>Staking is not active</h2><p>The interface will open only after the contract, custody model, reward source and legal gates are independently verified.</p><button type="button" disabled>Connect wallet — unavailable</button></section>
    <section className="spore-how"><span>PROPOSED FLOW</span><div>{steps.map(([id,title,copy]) => <article key={id}><b>{id}</b><h3>{title}</h3><p>{copy}</p></article>)}</div></section></>;
}

function FlowLine({ tone = "bronze" }: { tone?: "bronze" | "green" | "cream" }) {
  return <svg className={`spore-chart ${tone}`} viewBox="0 0 500 150" preserveAspectRatio="none" aria-hidden="true"><path d="M4 135 C54 128 63 105 102 110 S150 76 193 84 S252 48 300 61 S370 30 420 42 S463 18 496 24" /><path className="fill" d="M4 135 C54 128 63 105 102 110 S150 76 193 84 S252 48 300 61 S370 30 420 42 S463 18 496 24 L496 150 L4 150Z" /></svg>;
}

function Treasury() {
  return <><section className="spore-title"><span>VALUE ROUTING</span><h1>The <em>SPORE</em> treasury</h1><p>A transparent proposal for routing margin from completed compute. Nothing moves until real revenue, governance and audit trails exist.</p></section><PreviewNotice />
    <section className="spore-split"><h2>Verified compute margin enters one ledger, then follows explicit rules.</h2><div><article><Sprout /><strong>Network resilience</strong><p>Funds future worker incentives, reliability and operating reserves.</p></article><article><ShieldCheck /><strong>Participant distribution</strong><p>Only paid, attributable work can become a distribution event.</p></article></div></section>
    <section className="spore-metrics"><article className="primary"><span>RESERVE STATUS</span><strong>Not funded</strong><small>No treasury contract deployed</small></article><article><span>DISTRIBUTIONS</span><strong>0</strong><small>No reward events recorded</small></article><article><span>BUYBACKS</span><strong>Disabled</strong><small>No market operation configured</small></article></section>
    <section className="spore-chart-grid"><article><header><span>Illustrative reserve path</span><strong>Requires real revenue</strong></header><FlowLine tone="green" /></article><article><header><span>Illustrative distributions</span><strong>Requires verified work</strong></header><FlowLine tone="cream" /></article></section></>;
}

function Data() {
  const signals = [["Workers", "Connect to view"],["Native GPU", "No live feed"],["Browser", "No live feed"],["Image routes", "No live feed"],["Busy now", "—"],["SPORE price", "Not listed"]];
  return <><section className="spore-data-hero"><div><span>MYCELLIOS / DATA</span><h1>Network signals,<br/><em>without invented numbers.</em></h1><p>This preview shows the telemetry model. Production values will appear only from an authenticated network feed.</p></div><div className="spore-pulse" aria-hidden="true">{Array.from({length:18},(_,i)=><i key={i}/>)}</div></section><PreviewNotice />
    <section className="spore-live"><header><Activity /><h2>Live</h2><span>feed disconnected</span></header><div className="spore-signal-grid">{signals.map(([label,value])=><article key={label}><strong>{value}</strong><span><i />{label}</span></article>)}</div></section>
    <section className="spore-telemetry"><h2><Network /> Network telemetry</h2><div className="spore-chart-grid"><article><header><span>Jobs completed per day</span><strong>Awaiting source</strong></header><div className="spore-bars" aria-hidden="true">{[42,68,28,82,55,34,72,48,88,63,37,76,51,67,31,58,79,45].map((h,i)=><i key={i} style={{height:`${h}%`}} />)}</div></article><article><header><span>Useful compute recorded</span><strong>Schema preview</strong></header><FlowLine /></article></div></section>
    <section className="spore-provenance"><Database /><div><strong>Source before spectacle</strong><p>Every published metric will include its origin, sample window and last update. Price and market-cap panels stay absent until a real market exists.</p></div></section></>;
}

export function SporeStudio() {
  const path = window.location.pathname;
  return <main className="rb-page spore-page"><LandingHeader /><div className="spore-shell">{path.endsWith("/treasury") ? <Treasury /> : path.endsWith("/data") ? <Data /> : <Staking />}</div></main>;
}
