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
import { useEffect, useMemo, useRef, useState, type CSSProperties, type FormEvent } from "react";
import { loadAuthConfig, restoreAuthSession, type AuthSession } from "../auth";
import type { SavedStudioAgent } from "./studio-api";
import { loadStudioContinuity, saveStudioContinuity } from "./studio-continuity";
import {
  STUDIO_TEMPLATES,
  draftFromTemplate,
  previewReply,
  loadLocalStudioDraft,
  persistLocalStudioDraft,
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
  const [draft, setDraft] = useState<StudioDraft>(loadLocalStudioDraft);
  const [activeStep, setActiveStep] = useState<StudioStep>("identity");
  const [templateOpen, setTemplateOpen] = useState(true);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [knowledgeInput, setKnowledgeInput] = useState("");
  const [prompt, setPrompt] = useState("How would you introduce yourself to a first-time visitor?");
  const [conversation, setConversation] = useState<PreviewMessage[]>(() => [
    { role: "agent" as const, text: "I am ready to test. Change the draft and ask me something — this preview stays in your browser." },
  ]);
  const [saved, setSaved] = useState(true);
  const [storageFailed, setStorageFailed] = useState(false);
  const [session, setSession] = useState<AuthSession | null>(null);
  const [remoteAgent, setRemoteAgent] = useState<SavedStudioAgent | null>(null);
  const [remoteStatus, setRemoteStatus] = useState<string | null>(null);
  const [remoteError, setRemoteError] = useState(false);
  const [remoteMissing, setRemoteMissing] = useState(false);
  const [remoteBusy, setRemoteBusy] = useState(false);
  const publishing = useRef(false);
  const nameField = useRef<HTMLInputElement>(null);
  const roleField = useRef<HTMLInputElement>(null);
  const instructionsField = useRef<HTMLTextAreaElement>(null);
  const currentDraft = useRef(draft);
  currentDraft.current = draft;
  const completion = useMemo(() => studioCompletion(draft), [draft]);
  const continuity = session ? loadStudioContinuity(session.user.id) : null;
  const identityReady = Boolean(draft.name.trim() && draft.role.trim() && draft.instructions.trim());

  useEffect(() => {
    setSaved(false);
    const timeout = window.setTimeout(() => {
      const persisted = persistLocalStudioDraft(draft);
      setSaved(persisted);
      setStorageFailed(!persisted);
    }, 180);
    return () => window.clearTimeout(timeout);
  }, [draft]);

  useEffect(() => {
    const saveBeforeLeaving = () => { persistLocalStudioDraft(currentDraft.current); };
    window.addEventListener("pagehide", saveBeforeLeaving);
    return () => window.removeEventListener("pagehide", saveBeforeLeaving);
  }, []);

  useEffect(() => {
    let active = true;
    void loadAuthConfig(false).then(restoreAuthSession).then(async (value) => {
      if (!active) return;
      setSession(value);
      if (!value) return;
      const continuity = loadStudioContinuity(value.user.id);
      if (continuity.agentId) {
        try {
          const { loadStudioAgent } = await import("./studio-api");
          const agent = await loadStudioAgent(value, continuity.agentId);
          if (active) {
            setRemoteAgent(agent);
            if (continuity.publish) setRemoteStatus("A previous publication needs confirmation. Confirm it before sending new changes.");
          }
        } catch (error) {
          if (active) {
            setRemoteMissing(isMissingStudioAgent(error));
            setRemoteStatus(isMissingStudioAgent(error)
              ? "Your saved agent was not found in this account. You can start a new one from this local draft."
              : "Could not check your existing Studio agent. Retry to check before publishing.");
            setRemoteError(true);
          }
        }
      } else if (continuity.create && active) {
        setRemoteStatus("An earlier import was not confirmed. Retry to check it before publishing.");
      }
    }).catch(() => undefined);
    return () => { active = false; };
  }, []);

  function update(patch: Partial<StudioDraft>) {
    setDraft((current) => ({ ...current, ...patch }));
    setRemoteStatus(remoteAgent ? "Draft changed · publish to apply updates" : null);
    setRemoteError(false);
  }

  function chooseTemplate(id: StudioTemplateId) {
    setDraft(draftFromTemplate(id));
    setRemoteStatus(remoteAgent ? "Draft changed · publish to apply updates" : null);
    setRemoteError(false);
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
    if (publishing.current) return;
    const fresh = draftFromTemplate("concierge");
    setDraft(fresh);
    setConversation([{ role: "agent", text: "Draft reset. Choose a template or shape this identity from scratch." }]);
    setTemplateOpen(true);
    setRemoteStatus(remoteAgent ? "Local draft reset. Your published agent is unchanged." : null);
    setRemoteError(false);
  }

  function forgetMissingAgent() {
    if (!session) return;
    if (!saveStudioContinuity(session.user.id, {})) setStorageFailed(true);
    setRemoteAgent(null);
    setRemoteMissing(false);
    setRemoteError(false);
    setRemoteStatus("Saved link cleared. Your local draft is ready to import as a new agent.");
  }

  function completeIdentity() {
    setTemplateOpen(false);
    setActiveStep("identity");
    window.requestAnimationFrame(() => {
      if (!draft.name.trim()) nameField.current?.focus();
      else if (!draft.role.trim()) roleField.current?.focus();
      else instructionsField.current?.focus();
    });
  }

  async function saveAndPublish() {
    if (!session || publishing.current) return;
    if (!identityReady && !continuity?.publish) { completeIdentity(); return; }
    publishing.current = true;
    setRemoteBusy(true); setRemoteStatus(null); setRemoteError(false);
    try {
      const { importStudioDraft, loadStudioAgent, saveStudioDraft, publishStudioAgent, studioAgentMatchesDraft } = await import("./studio-api");
      const ownerId = session.user.id;
      let continuity = loadStudioContinuity(ownerId);
      let agent = remoteAgent;
      if (continuity.agentId && agent?.id !== continuity.agentId) {
        agent = await loadStudioAgent(session, continuity.agentId);
        setRemoteAgent(agent);
      }
      if (continuity.publish) {
        const pending = continuity.publish;
        if (!agent || agent.id !== pending.agentId) throw new Error("Could not verify the pending publication. Reload Studio and try again.");
        const confirmed = await publishStudioAgent(session, { ...agent, draftVersion: pending.draftVersion }, pending.channel, pending.key);
        continuity = { ...continuity, agentId: confirmed.agent.id, publish: undefined };
        if (!saveStudioContinuity(ownerId, continuity)) setStorageFailed(true);
        setRemoteAgent(confirmed.agent);
        setRemoteStatus("Previous publication confirmed. Review your draft before publishing further changes.");
        return;
      }
      if (!agent) {
        const pending = continuity.create ?? { key: `import-${crypto.randomUUID()}`, draft };
        continuity = { ...continuity, create: pending };
        if (!saveStudioContinuity(ownerId, continuity)) setStorageFailed(true);
        agent = await importStudioDraft(session, pending.draft, pending.key);
        continuity = { ...continuity, agentId: agent.id, create: undefined };
        if (!saveStudioContinuity(ownerId, continuity)) setStorageFailed(true);
        setRemoteAgent(agent);
      }
      if (!studioAgentMatchesDraft(agent, draft)) {
        agent = await saveStudioDraft(session, draft, agent, `import-${crypto.randomUUID()}`);
        setRemoteAgent(agent);
      }
      const pending = { key: `publish-${crypto.randomUUID()}`, agentId: agent.id, draftVersion: agent.draftVersion, channel: draft.channel };
      continuity = { ...continuity, publish: pending };
      if (!saveStudioContinuity(ownerId, continuity)) setStorageFailed(true);
      const result = await publishStudioAgent(session, agent, pending.channel, pending.key);
      if (!saveStudioContinuity(ownerId, { ...continuity, agentId: result.agent.id, publish: undefined })) setStorageFailed(true);
      setRemoteAgent(result.agent);
      setRemoteStatus(currentDraft.current !== draft ? "Earlier draft published. Your newer changes still need publication." : result.agent.operationalState === "waiting_for_capacity" ? "Published · waiting for compatible capacity" : `Published · ${result.agent.operationalState}`);
    } catch (error) {
      if (isMissingStudioAgent(error)) {
        setRemoteMissing(true);
        setRemoteStatus("Your saved agent was not found in this account. You can start a new one from this local draft.");
      } else setRemoteStatus(error instanceof Error ? error.message : "Studio could not save this agent.");
      setRemoteError(true);
    }
    finally { publishing.current = false; setRemoteBusy(false); }
  }

  return (
    <main className="studio-page">
      <header className="studio-topbar">
        <a className="studio-brand" href="/" aria-label="Mycellios home">
          <img src="/assets/brand/favicon.png" alt="" width={34} height={34} />
          <span>mycellios</span><i>studio</i>
        </a>
        <div className="studio-draft-status" aria-live="polite">
          <CircleDot />
          <span>{storageFailed ? "Browser storage unavailable · keep this tab open" : saved ? "Draft saved locally" : "Saving draft…"}</span>
        </div>
        <div className="studio-top-actions">
          <button type="button" className="studio-reset" disabled={remoteBusy} onClick={resetDraft}><RefreshCw />Reset</button>
          <a className="studio-login" href="/dashboard">Workspace <ArrowRight /></a>
        </div>
        <button className="studio-mobile-menu" type="button" aria-label={mobileNavOpen ? "Close studio navigation" : "Open studio navigation"} aria-expanded={mobileNavOpen} onClick={() => setMobileNavOpen((open) => !open)}>
          {mobileNavOpen ? <X /> : <Menu />}
        </button>
      </header>

      <div className="studio-shell">
        <aside className={`studio-rail ${mobileNavOpen ? "open" : ""}`} aria-label="Studio builder steps">
          <div className="studio-rail-heading">
            <span>Draft checklist</span>
            <strong>{completion}% filled</strong>
          </div>
          <div className="studio-progress" aria-label={`${completion}% of draft checklist filled`}><i style={{ width: `${completion}%` }} /></div>
          <nav>
            {steps.map(({ id, label, icon: Icon }, index) => (
              <button className={activeStep === id ? "active" : ""} type="button" onClick={() => { setActiveStep(id); setMobileNavOpen(false); }} key={id}>
                <span>{index + 1}</span><Icon /><b>{label}</b>{activeStep === id && <ChevronRight />}
              </button>
            ))}
          </nav>
          <div className="studio-rail-note">
            <Sparkles />
            <p><strong>Local draft</strong>Your draft stays in this browser. Publishing sends supported settings to your account.</p>
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
                <label><span>Name</span><input ref={nameField} value={draft.name} maxLength={48} onChange={(event) => update({ name: event.target.value })} /></label>
                <label><span>Role</span><input ref={roleField} value={draft.role} maxLength={100} onChange={(event) => update({ role: event.target.value })} /></label>
                <label className="studio-field-wide"><span>Personality and boundaries</span><textarea ref={instructionsField} value={draft.instructions} maxLength={600} onChange={(event) => update({ instructions: event.target.value })} /><small>{draft.instructions.length}/600 · Tell it how to behave, and what it must never invent.</small></label>
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
                <small>Labels stay in this browser. No source content is uploaded or included when you publish.</small>
              </form>
              <div className="studio-source-list">
                {draft.knowledge.map((source, index) => (
                  <div key={`${source}-${index}`}><span><BookOpen /><i>{String(index + 1).padStart(2, "0")}</i></span><p><strong>{source}</strong><small>Local label · no content connected</small></p><button type="button" aria-label={`Remove ${source}`} onClick={() => update({ knowledge: draft.knowledge.filter((_, itemIndex) => itemIndex !== index) })}><Trash2 /></button></div>
                ))}
                {draft.knowledge.length === 0 && <div className="studio-empty"><BookOpen /><p><strong>No source labels yet</strong><small>Add a name here to plan a source. No content is connected in this draft.</small></p></div>}
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
                <span>Review before publishing</span>
                <h2>{draft.name || "Untitled identity"} for {channels.find((channel) => channel.id === draft.channel)?.label}</h2>
                <ul><li className={identityReady ? "" : "waiting"}><Check />{identityReady ? "Identity and behavior configured" : "Add a name, role, and behavior"}</li><li className="waiting"><BookOpen />{draft.knowledge.length || "No"} local source label{draft.knowledge.length === 1 ? "" : "s"} · not published</li><li className={draft.tools.length ? "" : "waiting"}><Check />{draft.tools.length || "No"} tool{draft.tools.length === 1 ? "" : "s"} selected</li></ul>
                {remoteMissing && session
                  ? <button type="button" onClick={forgetMissingAgent}>Start a new agent from this draft <ArrowRight /></button>
                  : !identityReady && !continuity?.publish
                    ? <button type="button" onClick={completeIdentity}>Complete identity first <ArrowRight /></button>
                    : session
                  ? <button type="button" onClick={() => void saveAndPublish()} disabled={remoteBusy}>{remoteBusy ? "Checking and publishing…" : continuity?.publish ? "Confirm previous publication" : continuity?.create ? "Retry import and publish" : continuity?.agentId ? "Publish this revision" : "Import draft and publish"} <ExternalLink /></button>
                  : <a href="/dashboard">Sign in, then return to publish <ExternalLink /></a>}
                <small className={remoteError ? "error" : undefined} role={remoteError ? "alert" : undefined} aria-live="polite">{remoteStatus ?? (session ? "Import is explicit; your local draft remains unchanged." : "No agent has been deployed yet.")}</small>
              </div>
            </div>
          )}

          <div className="studio-step-nav">
            <button type="button" disabled={firstStep.id === activeStep} onClick={() => setActiveStep(previousStep(activeStep))}><ChevronLeft />Previous</button>
            <span>{steps.findIndex((step) => step.id === activeStep) + 1} of {steps.length}</span>
            <button className="primary" type="button" disabled={lastStep.id === activeStep} onClick={() => setActiveStep(nextStep(activeStep))}>Next <ChevronRight /></button>
          </div>
        </section>

        <aside className="studio-preview" aria-label="Local identity preview">
          <div className="studio-preview-heading">
            <div className="studio-avatar" aria-hidden="true"><span>{initials(draft.name)}</span><i style={{ "--signal": `${completion}%` } as CSSProperties} /></div>
            <div><span>Draft preview</span><strong>{draft.name || "Untitled identity"}</strong><small>{draft.role || "Add a role"}</small></div>
            <b><CircleDot />Draft</b>
          </div>
          <div className="studio-signals" aria-label="Configured identity signals">
            <span><Database />{memoryOptions.find((option) => option.id === draft.memoryMode)?.label}</span>
            <span><BookOpen />{draft.knowledge.length} local labels</span>
            <span><Wrench />{draft.tools.length} selected tools</span>
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
  return ({ identity: "Give it a point of view.", memory: "Choose what can persist.", knowledge: "Plan trusted sources.", tools: "Define what it can do.", launch: "Choose where it should live." })[step];
}

function stepDescription(step: StudioStep): string {
  return ({ identity: "Start from a working pattern, then make the voice and boundaries yours.", memory: "A persistent identity needs an explicit memory policy, not unlimited retention.", knowledge: "List sources to connect later. These labels remain local and provide no knowledge to the published agent.", tools: "Capabilities stay visible and controlled. Nothing runs until it is connected securely.", launch: "Choose a channel and review what will be published." })[step];
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

function isMissingStudioAgent(error: unknown): boolean {
  return error instanceof Error && "status" in error && error.status === 404;
}
