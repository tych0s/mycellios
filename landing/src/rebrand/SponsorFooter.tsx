import { ArrowUpRight } from "lucide-react";
import "./sponsor-footer.css";

export function SponsorFooter() {
  return (
    <section className="rb-sponsors" aria-labelledby="rb-sponsors-title">
      <div className="rb-shell rb-sponsors-inner">
        <div className="rb-sponsors-heading">
          <span id="rb-sponsors-title">Sponsors</span>
          <p>Support open infrastructure, physical testing and wider access to distributed AI.</p>
        </div>
        <a className="rb-sponsor-cta" href="mailto:hello@mycellios.com?subject=Sponsor%20Mycellios">
          <span className="rb-sponsor-mark" aria-hidden="true"><i /><i /><i /></span>
          <span><strong>Become a sponsor</strong><small>Help the network grow with verifiable support.</small></span>
          <ArrowUpRight aria-hidden="true" />
        </a>
      </div>
    </section>
  );
}
