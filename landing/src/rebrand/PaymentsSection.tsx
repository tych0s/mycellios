import { ArrowUpRight, BadgeCheck, HandCoins, ShieldCheck, Wallet } from "lucide-react";
import { useRef, type CSSProperties } from "react";
import { GutterFruiting } from "./GutterFruiting";
import { useInView } from "./use-motion";

/*
 * The exchange.
 *
 * Users need clear access; machine owners need a record of the work they ran.
 * This section shows plan access alongside signed stage attribution. Payouts
 * are not live, so the diagram stops at verified contribution evidence.
 *
 * The split is not decorative: 24 / 6 / 16 GB of one model is 52 / 13 / 35 % of
 * the job, so the bars are the memory shares from the diagram above. They show
 * how one route is attributed, not a public price or a guaranteed payout.
 */

const SHARES = [
  { name: "GPU", layers: "layers A–F", share: 52 },
  { name: "Laptop", layers: "layers G–N", share: 13 },
  { name: "PC", layers: "layers O–Z", share: 35 },
] as const;

const RULES = [
  { id: "01", icon: Wallet, title: "Choose your access", copy: "Check current plan availability and usage allowance in your account." },
  { id: "02", icon: BadgeCheck, title: "Meter real work", copy: "Each machine signs the stages it runs. Active plans show usage in your account." },
  { id: "03", icon: ShieldCheck, title: "Verify the work", copy: "Receipts are checked before contribution credit is recorded." },
] as const;

export function PaymentsSection() {
  const railRef = useRef<HTMLDivElement>(null);
  /* One switch drives the whole rail: coins start travelling, the machine cards
     land, the share bars grow and the receipts stamp — all off `is-live`, all
     transform/opacity, no re-render per frame. */
  const live = useInView(railRef, 0.32);

  return (
    <section className="rb-pay" id="payments" aria-labelledby="rb-pay-title">
      <GutterFruiting />
      <div className="rb-shell">
        <div className="rb-pay-head rb-reveal">
          <p className="rb-kicker"><i /><span>The exchange</span></p>
          <h2 id="rb-pay-title">Access for users.<br /><em>Proof for machine owners.</em></h2>
          <p>Mycellios Go is the network’s regular-access plan. Check its availability in your account. Accepted stages and signed receipts record each machine’s contribution.</p>
        </div>

        <div className={`rb-pay-rail ${live ? "is-live" : ""}`} ref={railRef}>
          <article className="rb-pay-card buyer">
            <span><Wallet />Your access</span>
            <strong>Mycellios Go</strong>
            <small>Current plans and usage in Account</small>
          </article>

          <div className="rb-pay-wire" aria-hidden="true">
            {[0, 1, 2].map((index) => <i key={index} style={{ "--i": index } as CSSProperties} />)}
          </div>

          {/* The three shares are a real list: on a phone the rail collapses and
              this is what carries the meaning of the section. */}
          <ol className="rb-pay-machines" aria-label="How one served route is attributed">
            {SHARES.map((machine, index) => (
              <li key={machine.name} style={{ "--i": index, "--share": machine.share } as CSSProperties}>
                <div className="rb-pay-machine-top">
                  <strong>{machine.name}</strong>
                  <small>{machine.layers}</small>
                </div>
                <div className="rb-pay-machine-sum">
                  <b>{machine.share}%</b>
                  <span>of route</span>
                </div>
                <div className="rb-pay-bar" aria-hidden="true"><i /></div>
                <em><BadgeCheck />receipt signed</em>
              </li>
            ))}
          </ol>

          <div className="rb-pay-wire" aria-hidden="true">
            {[0, 1, 2].map((index) => <i key={index} style={{ "--i": index } as CSSProperties} />)}
          </div>

          <article className="rb-pay-card seller">
            <span><HandCoins />Machine owners</span>
            <strong>Verified contribution</strong>
            <small>accepted stages · signed receipts</small>
          </article>
        </div>

        <p className="rb-pay-note">
          Illustrative attribution of one served route — the shares show the memory each machine lends to the model,
          not subscription pricing or a guaranteed payout.
          <a href="/spore/treasury">How value routes <ArrowUpRight /></a>
        </p>

        <div className="rb-pay-rules">
          {RULES.map((rule, index) => {
            const Icon = rule.icon;
            return (
              <article className="rb-rule rb-reveal" key={rule.id} style={{ "--i": index } as CSSProperties}>
                <div className="rb-rule-top"><Icon /><span>{rule.id}</span></div>
                <strong>{rule.title}</strong>
                <p>{rule.copy}</p>
              </article>
            );
          })}
        </div>

        <div className="rb-pay-actions rb-reveal">
          <a className="rb-pill rb-pill-dark" href="/account">Explore Mycellios Go <ArrowUpRight /></a>
          <a className="rb-pill rb-pill-ghost" href="/earn">Connect your machine <ArrowUpRight /></a>
        </div>

        <p className="rb-econ-caveat rb-reveal">Usage tokens are access credits, not $SPORE · contributor payouts and $SPORE are not live yet</p>
      </div>
    </section>
  );
}
