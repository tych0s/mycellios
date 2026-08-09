import { ChevronDown, Image as ImageIcon, Menu, X } from "lucide-react";
import { useState, type FormEvent } from "react";
import "./create-studio.css";

const styles = ["None", "Photo", "Cinematic", "Anime", "Digital Art", "3D"] as const;
const ratios = ["Square", "Portrait", "Landscape"] as const;

export function CreateStudio() {
  const [style, setStyle] = useState<(typeof styles)[number]>("Photo");
  const [ratio, setRatio] = useState<(typeof ratios)[number]>("Square");
  const [prompt, setPrompt] = useState("");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  function submit(event: FormEvent) {
    event.preventDefault();
    window.location.assign("/network");
  }

  return (
    <main className="create-page">
      <header className="create-header">
        <a className="create-brand" href="/" aria-label="Mycellios home"><img src="/assets/logos/logo.png" alt="" />mycellios</a>
        <nav aria-label="Main navigation">
          <a href="/network?view=inference">Chat</a><a className="active" href="/create" aria-current="page">Create</a><a href="/#how-it-works">How it works</a><a href="/#architecture">Architecture</a><a href="/#evidence">Evidence</a><a href="/blog">Blog</a>
        </nav>
        <a className="create-login" href="/network">Login</a>
        <button className="create-menu-button" type="button" aria-label={menuOpen ? "Close menu" : "Open menu"} aria-expanded={menuOpen} onClick={() => setMenuOpen((open) => !open)}>{menuOpen ? <X /> : <Menu />}</button>
        {menuOpen && <div className="create-mobile-nav"><a href="/network?view=inference">Chat</a><a className="active" href="/create">Create</a><a href="/#how-it-works">How it works</a><a href="/#architecture">Architecture</a><a href="/#evidence">Evidence</a><a href="/blog">Blog</a><a href="/network">Login</a></div>}
      </header>

      <section className="create-canvas" aria-live="polite">
        <div className="create-placeholder"><ImageIcon aria-hidden="true" /><strong>Your image appears here</strong><span>Describe it below and press Generate</span></div>
      </section>

      <form className="create-composer" onSubmit={submit}>
        <div className="create-controls">
          <div className="create-chip-group" aria-label="Image style">{styles.map((item) => <button type="button" className={style === item ? "selected" : ""} aria-pressed={style === item} onClick={() => setStyle(item)} key={item}>{item}</button>)}</div>
          <i aria-hidden="true" />
          <div className="create-chip-group create-ratios" aria-label="Aspect ratio">{ratios.map((item) => <button type="button" className={ratio === item ? "selected" : ""} aria-pressed={ratio === item} onClick={() => setRatio(item)} key={item}><span className={`ratio-icon ${item.toLowerCase()}`} />{item}</button>)}</div>
          <i aria-hidden="true" />
          <span className="create-age">●&nbsp; 18+</span>
        </div>
        <button className="create-advanced-toggle" type="button" aria-expanded={advancedOpen} onClick={() => setAdvancedOpen((open) => !open)}>Advanced <ChevronDown className={advancedOpen ? "open" : ""} /></button>
        {advancedOpen && <div className="create-advanced"><label>Images <select defaultValue="1"><option>1</option><option>2</option><option>4</option></select></label><label>Prompt guidance <select defaultValue="Balanced"><option>Subtle</option><option>Balanced</option><option>Strong</option></select></label></div>}
        <div className="create-prompt-row"><textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="Describe the image you want" aria-label="Image description" /><button type="submit">Log in to generate</button></div>
        <p><strong>20 credits ($0.20) per image</strong> · Enter to generate, Shift and Enter for a new line · saved in this browser only</p>
      </form>
    </main>
  );
}
