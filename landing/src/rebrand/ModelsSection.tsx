import { ArrowUpRight, Boxes } from "lucide-react";
import { useRef, type CSSProperties } from "react";
import { useInView } from "./use-motion";

/*
 * The models.
 *
 * This is an editorial shortlist, not runtime inventory. The Panel queries the
 * Hub and the adapter registry decides what can actually run. Keeping those two
 * facts separate prevents a newly released checkpoint from looking supported
 * merely because it appears here.
 */

const MODELS = [
  { name: "GPT-OSS", ref: "openai/gpt-oss-20b · 120b", href: "https://huggingface.co/openai/gpt-oss-20b", license: "Apache 2.0", scale: "20B / 120B", bar: 28, fit: "Local baseline", status: "local" },
  { name: "Gemma 4", ref: "google/gemma-4-12B · 31B", href: "https://huggingface.co/google/gemma-4-12B", license: "Apache 2.0", scale: "12B / 31B", bar: 34, fit: "Edge + vision", status: "local" },
  { name: "Qwen3.5 27B", ref: "Qwen/Qwen3.5-27B", href: "https://huggingface.co/Qwen/Qwen3.5-27B", license: "Apache 2.0", scale: "27B", bar: 31, fit: "Primary canary", status: "candidate" },
  { name: "Mistral Small 4", ref: "mistralai/Mistral-Small-4-119B-2603", href: "https://huggingface.co/mistralai/Mistral-Small-4-119B-2603", license: "Apache 2.0", scale: "119B · 6.5B active", bar: 53, fit: "Distributed candidate", status: "candidate" },
  { name: "DeepSeek V4", ref: "deepseek-ai/DeepSeek-V4-Flash · Pro", href: "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash", license: "MIT", scale: "284B / 1.6T", bar: 82, fit: "Long-context R&D", status: "research" },
  { name: "GLM-5.2", ref: "zai-org/GLM-5.2", href: "https://huggingface.co/zai-org/GLM-5.2", license: "MIT", scale: "753B", bar: 72, fit: "Coding + agents R&D", status: "research" },
  { name: "Kimi K3", ref: "moonshotai/Kimi-K3", href: "https://huggingface.co/moonshotai/Kimi-K3", license: "Kimi K3", scale: "2.8T · 104B active", bar: 100, fit: "Frontier benchmark", status: "benchmark" },
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
            A current shortlist from local-scale reasoning to frontier MoE systems. Mycellios evaluates each architecture
            independently: appearing here means it is worth integrating, not that the native adapter already ships.
          </p>
        </div>

        <div className={`rb-models-panel ${live ? "is-live" : ""}`} ref={panelRef}>
          <div className="rb-models-bar">
            <span className="rb-models-live"><i />August 2026 shortlist</span>
            <span>Official model cards · integration candidates</span>
          </div>

          <div className="rb-models-cols" aria-hidden="true">
            <span>Model</span><span>License</span><span>Scale</span><span>Mycellios fit</span>
          </div>

          <ol className="rb-models-list" aria-label="August 2026 open-model integration shortlist">
            {MODELS.map((model, index) => (
              <li key={model.ref} style={{ "--i": index, "--w": model.bar } as CSSProperties}>
                <div className="rb-model-id">
                  <strong><a href={model.href} target="_blank" rel="noreferrer">{model.name}<ArrowUpRight /></a></strong>
                  <small>{model.ref}</small>
                </div>
                <span className="rb-model-license">{model.license}</span>
                <div className="rb-model-size">
                  <b>{model.scale}</b>
                  <span className="rb-model-track" aria-hidden="true"><i /></span>
                </div>
                <div className={`rb-model-fit is-${model.status}`}>
                  <i aria-hidden="true" />
                  <em>{model.fit}</em>
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
          Scale is the published parameter count, not a memory or speed promise. Hardware fit still depends on
          precision, context, runtime support and the measured route selected by Mycellios.
        </p>

        <p className="rb-econ-caveat rb-reveal">Shortlist ≠ installed support · new families land only after an adapter and physical evidence ship</p>
      </div>
    </section>
  );
}
