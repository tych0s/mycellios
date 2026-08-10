import { ArrowUpRight, Boxes } from "lucide-react";
import { useRef, type CSSProperties } from "react";
import { useInView } from "./use-motion";

/*
 * The models.
 *
 * The runtime has no house model and no fixed catalogue: `model-catalog.ts`
 * queries Hugging Face live and accepts any checkpoint whose architecture has a
 * native adapter (llama, qwen3, qwen3_moe, glm4_moe today). So this section is
 * not a price list — it is one claim, drawn: the bigger the open model, the more
 * ordinary machines it takes, and nothing here caps out at what one box can hold.
 *
 * The proof is the last column. Each square is one machine, and they light up in
 * sequence as the row lands, so a 470 GB model visibly needs a row of boxes where
 * a 1.2 GB model needs one. That is the whole product in a single glance.
 *
 * Sizes are BF16 weight estimates (parameters × 2 bytes) — the same arithmetic
 * the coordinator uses in `estimateCatalogMemoryMiB` — not measured runtimes.
 */

const MODELS = [
  { name: "Qwen3 0.6B", ref: "Qwen/Qwen3-0.6B", family: "Qwen3", size: "1.2 GB", bar: 8, run: "One machine", boxes: 1 },
  { name: "Hermes 3 · Llama 3.2 3B", ref: "NousResearch/Hermes-3-Llama-3.2-3B", family: "Llama", size: "6.4 GB", bar: 17, run: "One machine", boxes: 1 },
  { name: "Qwen3 8B", ref: "Qwen/Qwen3-8B", family: "Qwen3", size: "16.4 GB", bar: 27, run: "One machine", boxes: 1 },
  { name: "Qwen3 30B-A3B", ref: "Qwen/Qwen3-30B-A3B", family: "Qwen3-MoE", size: "61 GB", bar: 45, run: "2–3 machines", boxes: 3 },
  { name: "GLM-4.5-Air", ref: "zai-org/GLM-4.5-Air", family: "GLM-4.5-MoE", size: "212 GB", bar: 72, run: "4–8 machines", boxes: 5 },
  { name: "Qwen3 235B-A22B", ref: "Qwen/Qwen3-235B-A22B", family: "Qwen3-MoE", size: "470 GB", bar: 100, run: "8+ machines", boxes: 7 },
] as const;

/* The adapter registry, named. These are the four families with a native
   pipeline-stage adapter shipped today; the list grows, which is why the copy
   says "today" and not "supported models". */
const FAMILIES = ["Llama", "Qwen3", "Qwen3-MoE", "GLM-4.5-MoE"] as const;

export function ModelsSection() {
  const panelRef = useRef<HTMLDivElement>(null);
  /* Same pattern as the payment rail: one class drives every row, bar and box,
     all transform/opacity, no re-render per frame. */
  const live = useInView(panelRef, 0.22);

  return (
    <section className="rb-models" id="models" aria-labelledby="rb-models-title">
      <div className="rb-shell">
        <div className="rb-models-head rb-reveal">
          <p className="rb-kicker"><i /><span>The models</span></p>
          {/* Two short lines: at this type size anything past ~26 characters
              wraps, and a ragged three-line headline is not a headline. */}
          <h2 id="rb-models-title">Any open model.<br /><em>Even the biggest ones.</em></h2>
          <p>
            No house model, no vendor lock. The network reads Hugging Face directly and runs any open checkpoint
            whose architecture has a native adapter — the small ones on one box, the large ones split across several.
          </p>
        </div>

        <div className={`rb-models-panel ${live ? "is-live" : ""}`} ref={panelRef}>
          <div className="rb-models-bar">
            <span className="rb-models-live"><i />Read live from Hugging Face</span>
            <span>Open weights · BF16 estimate</span>
          </div>

          <div className="rb-models-cols" aria-hidden="true">
            <span>Model</span><span>Family</span><span>Memory</span><span>Run mode</span>
          </div>

          <ol className="rb-models-list" aria-label="Open models and how many machines each one takes">
            {MODELS.map((model, index) => (
              <li key={model.ref} style={{ "--i": index, "--w": model.bar } as CSSProperties}>
                <div className="rb-model-id">
                  <strong>{model.name}</strong>
                  <small>{model.ref}</small>
                </div>
                <span className="rb-model-family">{model.family}</span>
                <div className="rb-model-size">
                  <b>{model.size}</b>
                  <span className="rb-model-track" aria-hidden="true"><i /></span>
                </div>
                <div className={`rb-model-run ${model.boxes > 1 ? "is-split" : ""}`}>
                  <em>{model.run}</em>
                  {/* One square per machine. The count is the argument. */}
                  <span className="rb-model-boxes" aria-hidden="true">
                    {Array.from({ length: model.boxes }, (_, box) => (
                      <i key={box} style={{ "--b": box } as CSSProperties} />
                    ))}
                  </span>
                </div>
              </li>
            ))}
          </ol>

          <div className="rb-models-foot">
            <Boxes />
            <span>Native adapters today: {FAMILIES.join(" · ")}</span>
            <a href="https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads" target="_blank" rel="noreferrer">
              Browse open models <ArrowUpRight />
            </a>
          </div>
        </div>

        <p className="rb-models-note rb-reveal">
          Sizes are BF16 weight estimates, not measured runtimes, and the machine counts are indicative.
          Single-machine models run today; the multi-machine split is in physical testing.
          <a href="#evidence">See what is built <ArrowUpRight /></a>
        </p>

        <p className="rb-econ-caveat rb-reveal">Model support is architecture-based · new families land as adapters ship</p>
      </div>
    </section>
  );
}
