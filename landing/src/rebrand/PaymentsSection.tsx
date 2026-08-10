import { ArrowUpRight, BadgeCheck, HandCoins, ShieldCheck, Wallet } from "lucide-react";
import { useRef, type CSSProperties } from "react";
import { useInView } from "./use-motion";

/*
 * The exchange.
 *
 * A distributed runtime is only interesting if both sides of it are: the buyer
 * wants one price for one answer, the machine owner wants to be paid for the
 * part it actually ran. This section is that single sentence, drawn — money
 * enters on the left, is split across the same three machines the scroll story
 * just introduced, and leaves on the right only against signed receipts.
 *
 * The split is not decorative: 24 / 6 / 16 GB of one model is 52 / 13 / 35 % of
 * the job, so the bars are the memory shares from the diagram above, restated
 * as payment. The figures illustrate the rule; they are not a live rate, and
 * the caveat under the rail says so.
 */

const SHARES = [
  { name: "GPU", layers: "layers A–F", share: 52, amount: "$0.0125" },
  { name: "Laptop", layers: "layers G–N", share: 13, amount: "$0.0031" },
  { name: "PC", layers: "layers O–Z", share: 35, amount: "$0.0084" },
] as const;

const RULES = [
  { id: "01", icon: Wallet, title: "Pay once", copy: "One request, one price. The buyer never negotiates with three machines." },
  { id: "02", icon: BadgeCheck, title: "Sign the work", copy: "Every machine signs a receipt for the exact stage it ran." },
  { id: "03", icon: ShieldCheck, title: "Settle on proof", copy: "Receipts are checked first. Work that fails verification is not paid." },
] as const;

export function PaymentsSection() {
  const railRef = useRef<HTMLDivElement>(null);
  /* One switch drives the whole rail: coins start travelling, the machine cards
     land, the share bars grow and the receipts stamp — all off `is-live`, all
     transform/opacity, no re-render per frame. */
  const live = useInView(railRef, 0.32);

  return (
    <section className="rb-pay" id="payments" aria-labelledby="rb-pay-title">
      <div className="rb-shell">
        <div className="rb-pay-head rb-reveal">
          <p className="rb-kicker"><i /><span>The exchange</span></p>
          <h2 id="rb-pay-title">Buyers pay for answers.<br /><em>Machines get paid for work.</em></h2>
          <p>One price per request, split between the machines that actually served it. Nothing is paid on trust — a stage is paid because it can be checked.</p>
        </div>

        <div className={`rb-pay-rail ${live ? "is-live" : ""}`} ref={railRef}>
          <article className="rb-pay-card buyer">
            <span><Wallet />Buyer</span>
            <strong>$0.0240</strong>
            <small>one request · charged once</small>
          </article>

          <div className="rb-pay-wire" aria-hidden="true">
            {[0, 1, 2].map((index) => <i key={index} style={{ "--i": index } as CSSProperties} />)}
          </div>

          {/* The three shares are a real list: on a phone the rail collapses and
              this is what carries the meaning of the section. */}
          <ol className="rb-pay-machines" aria-label="How one payment is split">
            {SHARES.map((machine, index) => (
              <li key={machine.name} style={{ "--i": index, "--share": machine.share } as CSSProperties}>
                <div className="rb-pay-machine-top">
                  <strong>{machine.name}</strong>
                  <small>{machine.layers}</small>
                </div>
                <div className="rb-pay-machine-sum">
                  <b>{machine.share}%</b>
                  <span>{machine.amount}</span>
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
            <strong>Paid per stage</strong>
            <small>settled in USDC · or in $SPORE later</small>
          </article>
        </div>

        <p className="rb-pay-note">
          Illustrative split of a single request — the shares are the memory each machine lends to the model,
          not a live rate.
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
          <a className="rb-pill rb-pill-dark" href="/network?view=inference">Buy compute <ArrowUpRight /></a>
          <a className="rb-pill rb-pill-ghost" href="/earn">Sell your idle machine <ArrowUpRight /></a>
        </div>

        <p className="rb-econ-caveat rb-reveal">Payment layer · design · no token is live and no payout is promised yet</p>
      </div>
    </section>
  );
}
