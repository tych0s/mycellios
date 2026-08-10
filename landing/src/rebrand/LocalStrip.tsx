import { ArrowUpRight, Terminal } from "lucide-react";
import { useRef } from "react";
import { useInView } from "./use-motion";

/*
 * A half section, deliberately.
 *
 * Between "here are the models" and "here are the three doors" there is one
 * short answer the reader needs and no full section deserves: you do not have to
 * join anything to try this. So this is a band, not a chapter — half the height,
 * no centred headline, the copy on the left and two unequal cards on the right.
 *
 * Both cards state something that is true in the repo rather than a posture:
 * the coordinator defaults to 127.0.0.1:8787 with API access off
 * (`src/core/config.ts`, `.env.example`), and sharing your hardware is a
 * separate switch that ships disabled (`src/desktop/settings.ts`). The footnote
 * says where the conversation is written, because claiming "we store nothing"
 * would be false — the local database keeps it.
 */

const CHIPS = ["No account", "No API key", "OpenAI-style /v1"] as const;

export function LocalStrip() {
  const bandRef = useRef<HTMLDivElement>(null);
  const live = useInView(bandRef, 0.35);

  return (
    <section className="rb-local" aria-labelledby="rb-local-title">
      <div className="rb-shell rb-local-inner" ref={bandRef}>
        <div className={`rb-local-copy ${live ? "is-live" : ""}`}>
          <p className="rb-kicker"><i /><span>On your machine</span></p>
          <h2 id="rb-local-title">Nothing to sign up for.<br /><em>Just a local address.</em></h2>
          <p>
            Run the coordinator on your own computer and it answers on loopback. No login, no key, no account —
            the network is something you point at, not something you join.
          </p>
        </div>

        <div className={`rb-local-cards ${live ? "is-live" : ""}`}>
          {/* The command types itself in: a bar slides off the text rather than
              the text being re-rendered, so the whole effect is one transform. */}
          <article className="rb-local-term">
            <span className="rb-local-term-top"><Terminal />Local endpoint</span>
            <p className="rb-local-cmd">
              <code>POST http://127.0.0.1:8787/v1/chat/completions</code>
              {/* The mask is the effect: it retracts to the right and uncovers
                  the line, and its ::before is a caret riding that edge — it
                  moves with the mask and stops, it does not blink. No
                  character-by-character state, no re-render, no idle loop. */}
              <i className="rb-local-mask" aria-hidden="true" />
            </p>
            <ul className="rb-local-chips">
              {CHIPS.map((chip) => <li key={chip}>{chip}</li>)}
            </ul>
          </article>

          {/* The one control that decides whether your machine works for anyone
              else. It is drawn in its shipped position: off. */}
          <article className="rb-local-switch">
            <span className="rb-local-knob" aria-hidden="true"><i /></span>
            <strong>Share your hardware</strong>
            <small>Off until you turn it on. Running locally never enlists your machine.</small>
          </article>
        </div>

        <p className="rb-local-foot">
          Local runs keep the conversation in a database file on that same machine — nothing is uploaded unless you
          point the app at a hosted coordinator.
          <a href="/downloads">Run it yourself <ArrowUpRight /></a>
        </p>
      </div>
    </section>
  );
}
