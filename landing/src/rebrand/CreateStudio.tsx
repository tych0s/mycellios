import {
  ArrowRight,
  BookOpen,
  Bot,
  Braces,
  Check,
  ChevronLeft,
  ChevronRight,
  CircleDot,
  Code2,
  Database,
  ExternalLink,
  Globe2,
  Menu,
  MessageCircle,
  Plus,
  RefreshCw,
  Send,
  Sparkles,
  Trash2,
  Wrench,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState, type CSSProperties, type FormEvent } from "react";
import { loadAuthConfig, restoreAuthSession, type AuthSession } from "../auth";
import { importStudioDraft, publishStudioAgent, type SavedStudioAgent } from "./studio-api";
import {
  STUDIO_STORAGE_KEY,
  STUDIO_TEMPLATES,
  draftFromTemplate,
  previewReply,
  restoreStudioDraft,
  studioCompletion,
  type MemoryMode,
  type StudioChannelId,
  type StudioDraft,
  type StudioTemplateId,
  type StudioToolId,
} from "./studio-model";
import "./create-studio.css";

const toolOptions: readonly { id: StudioToolId; label: string; detail: string; icon: typeof Globe2 }[] = [
  { id: "web", label: "Web research", detail: "Verify current information", icon: Globe2 },
  { id: "documents", label: "Knowledge", detail: "Search approved sources", icon: BookOpen },
  { id: "calculator", label: "Calculator", detail: "Handle exact arithmetic", icon: Braces },
  { id: "api", label: "API actions", detail: "Call allowlisted endpoints", icon: Code2 },
];

const channels: readonly { id: StudioChannelId; label: string; detail: string; icon: typeof Globe2 }[] = [
  { id: "web", label: "Web", detail: "Embeddable conversation", icon: Globe2 },
  { id: "telegram", label: "Telegram", detail: "Private or group bot", icon: Send },
  { id: "api", label: "API", detail: "OpenAI-compatible client", icon: Braces },
];

const memoryOptions: readonly { id: MemoryMode; label: string; detail: string }[] = [
  { id: "session", label: "This conversation", detail: "Forgets when the session ends" },
  { id: "approved", label: "Approved memories", detail: "Saves only reviewed facts" },
  { id: "continuous", label: "Continuous", detail: "Builds context across sessions" },
];

type StudioStep = "identity" | "memory" | "knowledge" | "tools" | "launch";
const steps: readonly { id: StudioStep; label: string; icon: typeof Bot }[] = [
  { id: "identity", label: "Identity", icon: Bot },
  { id: "memory", label: "Memory", icon: Database },
  { id: "knowledge", label: "Knowledge", icon: BookOpen },
  { id: "tools", label: "Tools", icon: Wrench },
  { id: "launch", label: "Launch", icon: Send },
];
const firstStep = steps[0]!;
const lastStep = steps[steps.length - 1]!;
type PreviewMessage = { role: "agent" | "user"; text: string };

export function CreateStudio() {
  const [draft, setDraft] = useState<StudioDraft>(() => restoreStudioDraft(window.localStorage.getItem(STUDIO_STORAGE_KEY)));
  const [activeStep, setActiveStep] = useState<StudioStep>("identity");
  const [templateOpen, setTemplateOpen] = useState(true);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [knowledgeInput, setKnowledgeInput] = useState("");
  const [prompt, setPrompt] = useState("How would you introduce yourself to a first-time visitor?");
  const [conversation, setConversation] = useState<PreviewMessage[]>(() => [
    { role: "agent" as const, text: "I am ready to test. Change the draft and ask me something — this preview stays in your browser." },
  ]);
  const [saved, setSaved] = useState(true);
  const [session, setSession] = useState<AuthSession | null>(null);
  const [remoteAgent, setRemoteAgent] = useState<SavedStudioAgent | null>(null);
  const [remoteStatus, setRemoteStatus] = useState<string | null>(null);
  const [remoteBusy, setRemoteBusy] = useState(false);
  const completion = useMemo(() => studioCompletion(draft), [draft]);

  useEffect(() => {
    setSaved(false);
    const timeout = window.setTimeout(() => {
      window.localStorage.setItem(STUDIO_STORAGE_KEY, JSON.stringify(draft));
      setSaved(true);
    }, 180);
    return () => window.clearTimeout(timeout);
  }, [draft]);

  useEffect(() => {
    let active = true;
    void loadAuthConfig().then(restoreAuthSession).then((value) => { if (active) setSession(value); }).catch(() => undefined);
    return () => { active = false; };
  }, []);

  function update(patch: Partial<StudioDraft>) {
    setDraft((current) => ({ ...current, ...patch }));
  }

  function chooseTemplate(id: StudioTemplateId) {
    setDraft(draftFromTemplate(id));
    setConversation([{ role: "agent", text: `Template loaded. I am ready to become ${draftFromTemplate(id).name}.` }]);
    setTemplateOpen(false);
  }

  function addKnowledge(event: FormEvent) {
    event.preventDefault();
    const source = knowledgeInput.trim();
    if (!source || draft.knowledge.includes(source) || draft.knowledge.length >= 6) return;
    update({ knowledge: [...draft.knowledge, source] });
    setKnowledgeInput("");
  }

  function toggleTool(id: StudioToolId) {
    update({ tools: draft.tools.includes(id) ? draft.tools.filter((tool) => tool !== id) : [...draft.tools, id] });
  }

  function testDraft(event: FormEvent) {
    event.preventDefault();
    if (!prompt.trim()) return;
    const question = prompt.trim();
    setConversation((current) => [
      ...current,
      { role: "user", text: question },
      { role: "agent", text: previewReply(draft, question) },
    ]);
    setPrompt("");
  }

  function resetDraft() {
    const fresh = draftFromTemplate("concierge");
    setDraft(fresh);
    setConversation([{ role: "agent", text: "Draft reset. Choose a template or shape this identity from scratch." }]);
    setTemplateOpen(true);
  }

  async function saveAndPublish() {
    if (!session || remoteBusy) return;
    setRemoteBusy(true); setRemoteStatus(null);
    try {
      const agent = remoteAgent ?? await importStudioDraft(session, draft, `import-${crypto.randomUUID()}`);
      setRemoteAgent(agent);
      const result = await publishStudioAgent(session, agent, draft.channel, `publish-${crypto.randomUUID()}`);
      setRemoteAgent(result.agent);
      setRemoteStatus(result.agent.operationalState === "waiting_for_capacity" ? "Published · waiting for compatible capacity" : `Published · ${result.agent.operationalState}`);
    } catch (error) { setRemoteStatus(error instanceof Error ? error.message : "Studio could not save this agent."); }
    finally { setRemoteBusy(false); }
  }

  return (
    <main className="studio-page">
      <header className="studio-topbar">
        <a className="studio-brand" href="/" aria-label="Mycellios home">
          <img src="/assets/logos/logo.png" alt="" />
          <span>mycellios</span><i>studio</i>
        </a>
        <div className="studio-draft-status" aria-live="polite">
          <CircleDot />
          <span>{saved ? "Draft saved locally" : "Saving draft…"}</span>
        </div>
        <div className="studio-top-actions">
          <button type="button" className="studio-reset" onClick={resetDraft}><RefreshCw />Reset</button>
          <a className="studio-login" href="/dashboard">Continue to workspace <ArrowRight /></a>
        </div>
        <button className="studio-mobile-menu" type="button" aria-label={mobileNavOpen ? "Close studio navigation" : "Open studio navigation"} aria-expanded={mobileNavOpen} onClick={() => setMobileNavOpen((open) => !open)}>
          {mobileNavOpen ? <X /> : <Menu />}
        </button>
      </header>

      <div className="studio-shell">
        <aside className={`studio-rail ${mobileNavOpen ? "open" : ""}`} aria-label="Studio builder steps">
          <div className="studio-rail-heading">
            <span>Build an identity</span>
            <strong>{completion}% ready</strong>
          </div>
          <div className="studio-progress" aria-label={`${completion}% ready`}><i style={{ width: `${completion}%` }} /></div>
          <nav>
            {steps.map(({ id, label, icon: Icon }, index) => (
              <button className={activeStep === id ? "active" : ""} type="button" onClick={() => { setActiveStep(id); setMobileNavOpen(false); }} key={id}>
                <span>{index + 1}</span><Icon /><b>{label}</b>{activeStep === id && <ChevronRight />}
              </button>
            ))}
          </nav>
          <div className="studio-rail-note">
            <Sparkles />
            <p><strong>Local draft</strong>Your configuration stays in this browser until you sign in and publish.</p>
          </div>
        </aside>

        <section className="studio-editor" aria-labelledby="studio-editor-title">
          <div className="studio-editor-heading">
            <div>
              <span className="studio-kicker">Mycellios Studio · {steps.find((step) => step.id === activeStep)?.label}</span>
              <h1 id="studio-editor-title">{stepTitle(activeStep)}</h1>
              <p>{stepDescription(activeStep)}</p>
            </div>
            {activeStep === "identity" && <button type="button" className="studio-text-action" onClick={() => setTemplateOpen((open) => !open)}>{templateOpen ? "Hide templates" : "Change template"}</button>}
          </div>

          {activeStep === "identity" && (
            <div className="studio-step-panel studio-identity-panel">
              {templateOpen && (
                <div className="studio-templates" aria-label="Identity templates">
                  {STUDIO_TEMPLATES.map((template) => (
                    <button type="button" className={draft.templateId === template.id ? "selected" : ""} onClick={() => chooseTemplate(template.id)} key={template.id}>
                      <span>{template.eyebrow}</span><strong>{template.title}</strong><p>{template.description}</p><i>{draft.templateId === template.id ? <Check /> : <ArrowRight />}</i>
                    </button>
                  ))}
                </div>
              )}
              <div className="studio-fields">
                <label><span>Name</span><input value={draft.name} maxLength={48} onChange={(event) => update({ name: event.target.value })} /></label>
                <label><span>Role</span><input value={draft.role} maxLength={100} onChange={(event) => update({ role: event.target.value })} /></label>
                <label className="studio-field-wide"><span>Personality and boundaries</span><textarea value={draft.instructions} maxLength={600} onChange={(event) => update({ instructions: event.target.value })} /><small>{draft.instructions.length}/600 · Tell it how to behave, and what it must never invent.</small></label>
              </div>
            </div>
          )}

          {activeStep === "memory" && (
            <div className="studio-step-panel studio-choice-list">
              {memoryOptions.map((option) => (
                <button type="button" className={draft.memoryMode === option.id ? "selected" : ""} onClick={() => update({ memoryMode: option.id })} key={option.id}>
                  <span className="studio-radio"><i /></span><p><strong>{option.label}</strong><small>{option.detail}</small></p>{draft.memoryMode === option.id && <Check />}
                </button>
              ))}
              <div className="studio-safety-note"><Database /><p><strong>Memory is policy, not a transcript dump.</strong>Approved mode is the safest default for a public identity: useful facts can persist without saving every conversation.</p></div>
            </div>
          )}

          {activeStep === "knowledge" && (
            <div className="studio-step-panel">
              <form className="studio-source-form" onSubmit={addKnowledge}>
                <label htmlFor="studio-source">Source label</label>
                <div><input id="studio-source" value={knowledgeInput} maxLength={64} placeholder="e.g. Product handbook" onChange={(event) => setKnowledgeInput(event.target.value)} /><button type="submit" disabled={!knowledgeInput.trim() || draft.knowledge.length >= 6}><Plus />Add source</button></div>
                <small>This MVP stores source names only. Files and URLs are connected after sign-in.</small>
              </form>
              <div className="studio-source-list">
                {draft.knowledge.map((source, index) => (
                  <div key={`${source}-${index}`}><span><BookOpen /><i>{String(index + 1).padStart(2, "0")}</i></span><p><strong>{source}</strong><small>Ready to connect · no content uploaded</small></p><button type="button" aria-label={`Remove ${source}`} onClick={() => update({ knowledge: draft.knowledge.filter((_, itemIndex) => itemIndex !== index) })}><Trash2 /></button></div>
                ))}
                {draft.knowledge.length === 0 && <div className="studio-empty"><BookOpen /><p><strong>No trusted source yet</strong><small>Add one label so the identity knows what it should be grounded in.</small></p></div>}
              </div>
            </div>
          )}

          {activeStep === "tools" && (
            <div className="studio-step-panel studio-tools-grid">
              {toolOptions.map(({ id, label, detail, icon: Icon }) => {
                const selected = draft.tools.includes(id);
                return <button type="button" className={selected ? "selected" : ""} aria-pressed={selected} onClick={() => toggleTool(id)} key={id}><Icon /><span><strong>{label}</strong><small>{detail}</small></span><i>{selected ? "On" : "Off"}</i></button>;
              })}
              <div className="studio-safety-note studio-tools-note"><Wrench /><p><strong>Tools are deny-by-default.</strong>Real actions require authentication, an allowlist, and explicit credentials. This draft never asks for secrets.</p></div>
            </div>
          )}

          {activeStep === "launch" && (
            <div className="studio-step-panel studio-launch-panel">
              <div className="studio-channel-list">
                {channels.map(({ id, label, detail, icon: Icon }) => (
                  <button type="button" className={draft.channel === id ? "selected" : ""} onClick={() => update({ channel: id })} key={id}><Icon /><span><strong>{label}</strong><small>{detail}</small></span>{draft.channel === id ? <Check /> : <ChevronRight />}</button>
                ))}
              </div>
              <div className="studio-launch-summary">
                <span>Ready to continue</span>
                <h2>{draft.name || "Untitled identity"} for {channels.find((channel) => channel.id === draft.channel)?.label}</h2>
                <ul><li><Check />Identity and behavior configured</li><li className={draft.knowledge.length ? "" : "waiting"}><Check />{draft.knowledge.length || "No"} knowledge source{draft.knowledge.length === 1 ? "" : "s"}</li><li className={draft.tools.length ? "" : "waiting"}><Check />{draft.tools.length || "No"} tool{draft.tools.length === 1 ? "" : "s"} enabled</li></ul>
                {session
                  ? <button type="button" onClick={() => void saveAndPublish()} disabled={remoteBusy}>{remoteBusy ? "Publishing…" : remoteAgent ? "Publish this revision" : "Import draft and publish"} <ExternalLink /></button>
                  : <a href="/network?view=overview">Sign in to connect and publish <ExternalLink /></a>}
                <small aria-live="polite">{remoteStatus ?? (session ? "Import is explicit; your local draft remains unchanged." : "No agent has been deployed yet.")}</small>
              </div>
            </div>
          )}

          <div className="studio-step-nav">
            <button type="button" disabled={firstStep.id === activeStep} onClick={() => setActiveStep(previousStep(activeStep))}><ChevronLeft />Previous</button>
            <span>{steps.findIndex((step) => step.id === activeStep) + 1} of {steps.length}</span>
            <button className="primary" type="button" disabled={lastStep.id === activeStep} onClick={() => setActiveStep(nextStep(activeStep))}>Next <ChevronRight /></button>
          </div>
        </section>

        <aside className="studio-preview" aria-label="Live identity preview">
          <div className="studio-preview-heading">
            <div className="studio-avatar" aria-hidden="true"><span>{initials(draft.name)}</span><i style={{ "--signal": `${completion}%` } as CSSProperties} /></div>
            <div><span>Live preview</span><strong>{draft.name || "Untitled identity"}</strong><small>{draft.role || "Add a role"}</small></div>
            <b><CircleDot />Draft</b>
          </div>
          <div className="studio-signals" aria-label="Configured identity signals">
            <span><Database />{memoryOptions.find((option) => option.id === draft.memoryMode)?.label}</span>
            <span><BookOpen />{draft.knowledge.length} sources</span>
            <span><Wrench />{draft.tools.length} tools</span>
          </div>
          <div className="studio-conversation" aria-live="polite">
            {conversation.slice(-5).map((message, index) => (
              <div className={message.role} key={`${message.role}-${index}`}><span>{message.role === "agent" ? initials(draft.name) : "You"}</span><p>{message.text}</p></div>
            ))}
          </div>
          <form className="studio-test-form" onSubmit={testDraft}>
            <label htmlFor="studio-test">Test this identity</label>
            <div><textarea id="studio-test" rows={3} value={prompt} placeholder={`Ask ${draft.name || "this identity"} something…`} onChange={(event) => setPrompt(event.target.value)} /><button type="submit" disabled={!prompt.trim()} aria-label="Send test message"><ArrowRight /></button></div>
            <small><MessageCircle />Preview only · no model or external tool is called</small>
          </form>
        </aside>
      </div>
    </main>
  );
}

function stepTitle(step: StudioStep): string {
  return ({ identity: "Give it a point of view.", memory: "Choose what can persist.", knowledge: "Ground it in trusted context.", tools: "Define what it can do.", launch: "Choose where it should live." })[step];
}

function stepDescription(step: StudioStep): string {
  return ({ identity: "Start from a working pattern, then make the voice and boundaries yours.", memory: "A persistent identity needs an explicit memory policy, not unlimited retention.", knowledge: "Name the sources this identity should trust before connecting their contents.", tools: "Capabilities stay visible and controlled. Nothing runs until it is connected securely.", launch: "Select a first channel and inspect exactly what is ready before publishing." })[step];
}

function previousStep(step: StudioStep): StudioStep {
  return steps[Math.max(0, steps.findIndex((item) => item.id === step) - 1)]!.id;
}

function nextStep(step: StudioStep): StudioStep {
  return steps[Math.min(steps.length - 1, steps.findIndex((item) => item.id === step) + 1)]!.id;
}

function initials(name: string): string {
  const value = name.trim();
  return value ? value.slice(0, 2).toUpperCase() : "AI";
}
