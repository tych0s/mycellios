import { ArrowUpRight } from "lucide-react";
import "./questions.css";

/*
 * The objections, answered before the download ask.
 *
 * This sits between the evidence board and the install section on purpose: a
 * reader who has just been shown what is and is not built has doubts, and the
 * page loses them if it answers those doubts after asking for the install
 * rather than before it.
 *
 * Two rules hold for every answer here:
 *
 *  - it is checkable in the repo. The 4 GB starting offer and the shipped-off
 *    contribution switch are `DEFAULT_DESKTOP_SETTINGS` in
 *    `src/desktop/settings.ts`; the loopback default is `src/core/config.ts`;
 *    stage recovery is the row the evidence board above marks LAB.
 *  - it may lose the reader. "Stages are not end-to-end encrypted" and "there is
 *    no token" cost conversions, and both stay in, because the section is only
 *    worth having if the uncomfortable question is the one it takes.
 *
 * Markup is native <details>: it opens with no JavaScript, is keyboard- and
 * screen-reader-correct for free, and every answer is in the server-rendered
 * HTML, so a crawler reads the same page a visitor does.
 */

const QUESTIONS = [
  {
    id: "01",
    q: "Do I need a powerful GPU to take part?",
    a: "No. The scheduler places stages by what a machine actually has, so a 6 GB laptop can hold a slice of the same model as a 24 GB card. You decide how much memory to lend — the desktop app starts at 4 GB and never takes more than you set.",
  },
  {
    id: "02",
    q: "What actually runs on my machine if I join?",
    a: "One stage of a model, and nothing else. The worker executes model layers, not programs sent by other users. Contribution ships switched off, you choose the memory it may use, and you can stop it at any moment.",
  },
  {
    id: "03",
    q: "How is this different from running a model locally?",
    a: "A local runtime stops at what one box can hold. Here the weights are cut into stages spread over several machines, so the ceiling is the memory of the group rather than of your card. The endpoint you call is OpenAI-shaped in both cases.",
  },
  {
    id: "04",
    q: "What happens if a machine disappears mid-answer?",
    a: "The stage is re-routed to another machine and the response continues instead of starting over. Single-machine serving runs today; that recovery path is still in the lab, which is why the board above marks it LAB and not READY.",
  },
  {
    id: "05",
    q: "Who can see what I send?",
    a: "Point the app at your own computer and the conversation stays on it, in a local database file. On the public network your request is served by machines you do not own, and stages are not end-to-end encrypted yet — until that ships, treat it as you would any hosted API.",
  },
  {
    id: "06",
    q: "Is there a token? Can I earn today?",
    a: "No. Nothing is live: no token, no staking, no payouts. USDC settlement and $SPORE are published design work so they can be argued with before they exist — they are not an offer, and no figure on this page is a rate.",
  },
] as const;

export function QuestionsSection() {
  return (
    <section className="rb-quest" id="questions" aria-labelledby="rb-quest-title">
      <div className="rb-shell rb-quest-inner">
        {/* The heading stays with the reader while the answers scroll past it,
            so the column never loses what it is answering. */}
        <div className="rb-quest-head rb-reveal">
          <p className="rb-kicker"><i /><span>Straight answers</span></p>
          <h2 id="rb-quest-title">The awkward<br /><em>questions.</em></h2>
          <p>The ones worth asking before installing anything. If the honest answer is “not yet”, that is the answer printed here.</p>
        </div>

        <div className="rb-quest-list rb-reveal">
          {QUESTIONS.map((item) => (
            <details className="rb-quest-item" key={item.id}>
              <summary>
                <span className="rb-quest-no">{item.id}</span>
                <span className="rb-quest-q">{item.q}</span>
                {/* Two bars: the upright one rotates flat on open, so the plus
                    becomes a minus in a single transform. */}
                <i className="rb-quest-sign" aria-hidden="true"><b /><b /></i>
              </summary>
              <div className="rb-quest-body"><p>{item.a}</p></div>
            </details>
          ))}

          <p className="rb-quest-foot">
            Still unsure? The assistant in the corner answers from the same documentation.
            <a href="/docs">Read the docs <ArrowUpRight /></a>
          </p>
        </div>
      </div>
    </section>
  );
}
