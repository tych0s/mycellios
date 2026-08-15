import { Check, Cpu, Laptop, MemoryStick, Sparkles } from "lucide-react";
import { useRef, type CSSProperties } from "react";
import { useStoryProgress } from "./use-motion";

/*
 * The one explainer on the page.
 *
 * Everything the landing needs to say about the product is a single sentence —
 * *one model, split across ordinary machines, returning one answer* — so it is
 * told once, as a scrubbed scroll story, instead of being restated by an
 * explainer, a flow section, an architecture diagram and a pipeline card.
 *
 * `--p` runs 0→1 across the pinned section and CSS interpolates every mark from
 * it. React only re-renders when the beat changes, which is at most four times
 * for the whole section.
 *
 * The story plays itself once pinned rather than waiting to be cranked. Four
 * beats behind three viewports of wheel travel made the reader work a rowing
 * machine for four sentences; now the section holds them for a few seconds,
 * tells its one sentence, and lets go. Scrolling still scrubs ahead for anyone
 * who would rather not wait.
 */

/* `step` is the name the beat carries in the rail under the diagram. Four
   anonymous dashes told the reader how far they had scrolled; the same four
   marks labelled tell them what the machine is doing, which is the whole point
   of the section. */
const BEATS = [
  { id: "01", step: "Request", title: "One question", copy: "You send a normal request. Nothing about your app changes." },
  { id: "02", step: "Split model", title: "The model splits", copy: "It is cut into stages small enough for ordinary machines." },
  { id: "03", step: "Run stages", title: "Machines take a part", copy: "A GPU, a laptop, a spare PC. Each one runs a single stage." },
  { id: "04", step: "Stream result", title: "One answer returns", copy: "The stages stream back as a single, continuous response." },
] as const;

/* `layers` makes the split concrete: three cards reading 24/6/16 GB show three
   machines, but 0–6 / 7–13 / 14–20 show one model cut across them. The uneven
   capacities are deliberate — a 6 GB laptop next to a 24 GB GPU is the whole
   claim: ordinary machines, not matched hardware. */
const DEVICES = [
  { icon: Cpu, name: "GPU", detail: "24 GB", layers: "layers 0–6" },
  { icon: Laptop, name: "Laptop", detail: "6 GB", layers: "layers 7–13" },
  { icon: MemoryStick, name: "PC", detail: "16 GB", layers: "layers 14–20" },
] as const;

export function ScrollStory() {
  const sectionRef = useRef<HTMLElement>(null);
  const beat = useStoryProgress(sectionRef, BEATS.length);
  const active = BEATS[beat] ?? BEATS[0];

  return (
    <section className="rb-story" id="how-it-works" ref={sectionRef} aria-labelledby="rb-story-title">
      <div className={`rb-story-sticky beat-${beat}`}>
        <div className="rb-shell rb-story-inner">
          <header className="rb-story-head">
            <p className="rb-kicker"><i /><span>How it works</span></p>
            <h2 id="rb-story-title">One model.<br /><em>Many machines.</em></h2>
          </header>

          <div className="rb-stage" aria-hidden="true">
            <div className="rb-stage-ask"><Sparkles /><span>Your request</span></div>

            <div className="rb-stage-model">
              {DEVICES.map((device, index) => (
                <i key={device.name} className={`shard-${index}`} style={{ "--i": index } as CSSProperties} />
              ))}
            </div>

            <div className="rb-stage-devices">
              {DEVICES.map((device, index) => {
                const Icon = device.icon;
                return (
                  <div key={device.name} className={`rb-stage-device device-${index}`} style={{ "--i": index } as CSSProperties}>
                    <Icon />
                    <strong>{device.name}</strong>
                    <small>{device.detail}</small>
                    {/* The stage number is wrapped so the phone layout can drop
                        it and keep the layer range. */}
                    <b><span className="stage-no">stage {index + 1} · </span>{device.layers}</b>
                  </div>
                );
              })}
            </div>

            {/* preserveAspectRatio="none" lets the wires stretch with the stage,
                so every coordinate here is really a percentage of the stage box
                (920×380 at desktop). x: 90/900 ≈ the right edge of the ask
                circle, 337 and 562 the edges of the 230px card column, 816 the
                left edge of the answer circle. y: 109/210/311 = 26%/50%/74%,
                which is where `.rb-stage-devices` puts the three card centres. */}
            <svg className="rb-stage-wires" viewBox="0 0 900 420" preserveAspectRatio="none">
              <path className="wire wire-0" d="M90 210C200 210 230 109 337 109" />
              <path className="wire wire-1" d="M90 210C200 210 230 210 337 210" />
              <path className="wire wire-2" d="M90 210C200 210 230 311 337 311" />
              <path className="wire wire-out wire-out-0" d="M562 109C660 109 700 210 816 210" />
              <path className="wire wire-out wire-out-1" d="M562 210C660 210 700 210 816 210" />
              <path className="wire wire-out wire-out-2" d="M562 311C660 311 700 210 816 210" />
            </svg>

            <div className="rb-stage-answer"><Check /><span>One answer</span></div>
          </div>

          <footer className="rb-story-foot">
            <div className="rb-story-copy" key={active.id}>
              <span>{active.id}</span>
              <strong>{active.title}</strong>
              <p>{active.copy}</p>
            </div>
            {/* Not aria-hidden: unlike the four dashes it replaces, this rail
                carries the names of the four steps, and the sticky section only
                ever exposes one beat's copy at a time. */}
            <ol className="rb-story-track" aria-label="Steps in a request">
              {BEATS.map((item, index) => (
                <li key={item.id} className={index <= beat ? "is-on" : ""}>
                  <i aria-hidden="true" />
                  <span>{item.step}</span>
                </li>
              ))}
            </ol>
          </footer>
        </div>
      </div>

      {/* Static equivalent for reduced-motion readers and for crawlers: the same
          four beats, with no scroll dependency. */}
      <ol className="rb-story-fallback rb-shell">
        {BEATS.map((item) => (
          <li key={item.id}><span>{item.id}</span><div><strong>{item.title}</strong><p>{item.copy}</p></div></li>
        ))}
      </ol>
    </section>
  );
}
