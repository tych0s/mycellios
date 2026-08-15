/* <mycelium-field> — scroll-grown mycelium network across the whole page.
   Two layers: a persistent full-page growth canvas + a fixed viewport canvas for travelling glints.
   Attributes: accent | origin-x | origin-y (fractions of the host box) | density | motion */
/*
 * Vendored from the design reference. Zero dependencies: this is Canvas2D.
 *
 * The one substantive change is `mycellios: dim`. The reference page is
 * #0a0908 from top to bottom, so it composites every filament with
 * `globalCompositeOperation = 'lighter'` — additive light on black. This
 * landing alternates dark bands (hero, install, closing) with ivory and paper
 * ones, and *adding* light to a near-white surface is a no-op: the network
 * would simply vanish for two thirds of the page. So the host element reports
 * how lit the surface behind it is, and over light bands the field switches to
 * `source-over` with the bronze ink darkened instead of added.
 *
 * Also exported as an idempotent function rather than an IIFE, so React's
 * StrictMode double-mount cannot race `customElements.define`.
 */
export function defineMyceliumField() {
  if (window.customElements && customElements.get('mycelium-field')) return;

  const mulberry32 = (a) => () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  class MyceliumField extends HTMLElement {
    static get observedAttributes() { return ['accent', 'progress', 'density', 'motion']; }

    connectedCallback() {
      this._dead = false;
      if (this._booted) {
        cancelAnimationFrame(this._raf);
        if (this._ro) this._ro.observe(this);
        this.tick();
        return;
      }
      this._booted = true;
      this.style.cssText += ';display:block;position:absolute;inset:0;width:100%;height:100%;pointer-events:none;';

      this.growth = document.createElement('canvas');
      this.growth.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;display:block;';
      this.glint = document.createElement('canvas');
      this.glint.style.cssText = 'position:fixed;inset:0;width:100vw;height:100vh;display:block;pointer-events:none;';
      this.appendChild(this.growth);
      this.appendChild(this.glint);

      this.accent = this.getAttribute('accent') || '#C9976A';
      this.motion = this.getAttribute('motion') || 'full';
      this.reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
      this._prog = 0; this._shown = 0; this._cursor = 0;
      this.sprite = this.makeSprite();
      this.sporeSprite = this.makeSporeSprite();
      this.spores = this.makeSpores();

      this._ro = new ResizeObserver(() => this.layout());
      this._ro.observe(this);
      this._onScroll = () => { this._dirty = true; };
      addEventListener('scroll', this._onScroll, { passive: true });
      addEventListener('resize', this._onScroll);
      this._dirty = true;
      this.layout();
      this.tick();
    }

    attributeChangedCallback(n, _o, v) {
      if (n === 'progress') this.setProgress(parseFloat(v));
      if (n === 'motion') this.motion = v || 'full';
    }

    disconnectedCallback() {
      this._dead = true;
      cancelAnimationFrame(this._raf);
      removeEventListener('scroll', this._onScroll);
      removeEventListener('resize', this._onScroll);
      if (this._ro) this._ro.disconnect();
    }

    selfProgress() {
      const vh = innerHeight;
      const rel = Math.max(0, this.effectiveScrollY() - (this._pageTop || 0));
      const range = this.storyRange();
      const H = Math.max(1, (this._h || 1) - (range?.travel || 0));
      const g = Math.min(1, rel / (vh * 0.45));
      const gate = g * g * (3 - 2 * g);
      const reach = (rel + vh * 0.75) / Math.max(1, H - vh * 0.25);
      return Math.max(0, Math.min(1, gate * (0.055 + 0.96 * reach)));
    }

    storyRange() {
      const story = this._story || (this._story = document.querySelector('.rb-story'));
      if (!story) return null;
      const top = (window.scrollY || 0) + story.getBoundingClientRect().top;
      const travel = Math.max(0, story.offsetHeight - innerHeight);
      return { top, end: top + travel, travel };
    }

    storyIsPinned() {
      const range = this.storyRange();
      const y = window.scrollY || 0;
      return !!range && y >= range.top && y < range.end;
    }

    placeGrowthCanvas() {
      const range = this.storyRange();
      const y = window.scrollY || 0;
      const pinned = !!range && y >= range.top && y < range.end;
      const mode = pinned ? 'fixed' : (range && y >= range.end ? 'after' : 'before');
      if (mode === this._growthMode) return;
      this._growthMode = mode;

      if (mode === 'fixed') {
        /* A fixed, document-sized canvas keeps the exact same viewport slice
           on screen while the story consumes virtual scroll. Unlike an RAF
           counter-transform, this runs in the browser compositor and cannot
           lag one frame behind the scroll position. */
        const pageTop = this._pageTop || 0;
        this.growth.style.position = 'fixed';
        this.growth.style.inset = 'auto';
        this.growth.style.left = '0';
        this.growth.style.top = `${-(range.top - pageTop)}px`;
        this.growth.style.width = `${this._w}px`;
        this.growth.style.height = `${this._h}px`;
        this.growth.style.transform = 'none';
        return;
      }

      this.growth.style.position = 'absolute';
      this.growth.style.inset = '0';
      this.growth.style.width = '100%';
      this.growth.style.height = '100%';
      this.growth.style.transform = `translate3d(0,${mode === 'after' ? range.travel : 0}px,0)`;
    }

    effectiveScrollY() {
      const y = window.scrollY || 0;
      const range = this.storyRange();
      if (!range || y <= range.top) return y;
      if (y < range.end) return range.top;
      return y - range.travel;
    }

    /*
     * Viewport y of the growth canvas's own origin — the one number that maps
     * a filament's canvas coordinate onto the screen.
     *
     * `placeGrowthCanvas` moves that canvas three different ways depending on
     * where the story is, so reading the host's box is only correct in one of
     * the three. This mirrors those three cases, and everything drawn on the
     * fixed glint canvas goes through it: without that, marks meant to sit on
     * a filament detach from it by the story's whole travel for the entire
     * lower half of the page.
     */
    canvasTop() {
      const hostTop = this.getBoundingClientRect().top;
      const range = this.storyRange();
      if (!range) return hostTop;
      const y = window.scrollY || 0;
      if (y >= range.end) return hostTop + range.travel;
      if (y >= range.top) return -(range.top - (this._pageTop || 0));
      return hostTop;
    }

    lockOrigin() {
      const sel = this.getAttribute('origin-from') || 'mushroom-stage';
      const src = document.querySelector(sel);
      if (!src) return false;
      const b = src.getBoundingClientRect();
      if (b.height < 20) return false;
      const r = this.getBoundingClientRect();
      /* Start below the rendered host, not four percent inside it. The hero
         already places that host 8px below the viewport edge, so b.bottom is a
         naturally hidden junction: the roots emerge on the next surface
         without painting a bronze starter segment over the visible stem. */
      this.setOrigin(b.left - r.left + b.width * 0.5, b.bottom - r.top);
      return true;
    }

    setProgress(p) { this._prog = Math.max(0, Math.min(1, p || 0)); }

    setOrigin(x, y) {
      if (this._ox != null && Math.abs(x - this._ox) < 6 && Math.abs(y - this._oy) < 6) return;
      this._ox = x; this._oy = y;
      if (!this._w) return;
      this.build(this._w, this._h);
      this._cursor = 0; this._nodeCursor = 0; this._shown = 0;
      this.gctx.clearRect(0, 0, this._w, this._h);
    }

    rgba(a) {
      const h = this.accent.replace('#', '');
      const r = parseInt(h.slice(0, 2), 16), g = parseInt(h.slice(2, 4), 16), b = parseInt(h.slice(4, 6), 16);
      return `rgba(${r},${g},${b},${a})`;
    }

    makeSprite() {
      const c = document.createElement('canvas');
      c.width = c.height = 64;
      const x = c.getContext('2d');
      const g = x.createRadialGradient(32, 32, 0, 32, 32, 32);
      g.addColorStop(0, 'rgba(255,244,230,0.95)');
      g.addColorStop(0.25, this.rgba(0.5));
      g.addColorStop(1, 'rgba(0,0,0,0)');
      x.fillStyle = g;
      x.fillRect(0, 0, 64, 64);
      return c;
    }

    makeSporeSprite() {
      const c = document.createElement('canvas');
      c.width = c.height = 64;
      const x = c.getContext('2d');
      const g = x.createRadialGradient(32, 32, 0, 32, 32, 32);
      g.addColorStop(0, this.rgba(0.94));
      g.addColorStop(0.12, this.rgba(0.58));
      g.addColorStop(0.46, this.rgba(0.16));
      g.addColorStop(1, 'rgba(0,0,0,0)');
      x.fillStyle = g;
      x.fillRect(0, 0, 64, 64);
      return c;
    }

    makeSpores() {
      const rnd = mulberry32(20260817);
      const count = 56;
      return Array.from({ length: count }, (_, i) => {
        /* Stratification guarantees that every part of the viewport receives
           spores instead of leaving the reading column accidentally empty. */
        const x = (i + 0.18 + rnd() * 0.64) / count;
        return {
          x,
          y: rnd() * 1.16,
          speed: 0.011 + rnd() * 0.012,
          size: 18 + rnd() * 24,
          alpha: 0.4 + rnd() * 0.34,
          drift: 0.004 + rnd() * 0.014,
          phase: rnd() * Math.PI * 2
        };
      });
    }

    layout() {
      const r = this.getBoundingClientRect();
      const w = Math.max(2, Math.round(r.width)), h = Math.max(2, Math.round(r.height));
      if (w === this._w && h === this._h) return;
      this._w = w; this._h = h;
      let dpr = Math.min(window.devicePixelRatio || 1, 1.25);
      const budget = 8.5e6;
      if (w * h * dpr * dpr > budget) dpr = Math.max(0.75, Math.sqrt(budget / (w * h)));
      this.dpr = dpr;
      this._pageTop = (window.scrollY || 0) + r.top;
      this.growth.width = Math.round(w * dpr);
      this.growth.height = Math.round(h * dpr);
      this.gctx = this.growth.getContext('2d');
      this.gctx.scale(dpr, dpr);
      this.gctx.lineCap = 'round';

      const vw = Math.max(2, innerWidth), vh = Math.max(2, innerHeight);
      const gdpr = Math.min(window.devicePixelRatio || 1, 1.5);
      this.glint.width = Math.round(vw * gdpr);
      this.glint.height = Math.round(vh * gdpr);
      this.lctx = this.glint.getContext('2d');
      this.lctx.scale(gdpr, gdpr);
      this.vw = vw; this.vh = vh;

      this.build(w, h);
      this._cursor = 0;
      this._nodeCursor = 0;
      this._shown = 0;
      this.gctx.clearRect(0, 0, w, h);
      this._originTries = 0;
      this._originLocked = false;
      this._growthMode = null;
    }

    build(W, H) {
      const rnd = mulberry32(20260811);
      const segs = [], nodes = [], trunks = [];
      const density = parseFloat(this.getAttribute('density')) || 1;
      const ox = this._ox != null ? this._ox : (parseFloat(this.getAttribute('origin-x')) || 0.34) * W;
      const oy = this._oy != null ? this._oy : (parseFloat(this.getAttribute('origin-y')) || 0.62) * H;
      const corridorStart = oy + 18;
      const corridorLeft = W * 0.12;
      const corridorRight = W * 0.88;
      /* Below the short root flare, the centre belongs to the message. Painting the
         canvas behind transparent content does not stop a trunk showing
         through type, so only the two edge rails are allowed to receive
         visible segments after the physical connection beneath the stem has
         become clear. */
      const readingZoneStart = oy + Math.min(140, Math.max(96, (this.vh || 900) * 0.14));
      /* Keep the central reading column clear, but give the organic left and
         right runners enough room to arc through the story section and meet
         the next generation. A narrow 14% rail made that union look cut off. */
      const railInset = W * 0.32;
      const clearsReadingZone = (x0, x1, y1) => (
        y1 <= readingZoneStart ||
        (x0 <= railInset && x1 <= railInset) ||
        (x0 >= W - railInset && x1 >= W - railInset)
      );
      /* A filament is a path, not a scatter of marks: rejecting the segments
         that happen to pass over the centre column does not hide the filament,
         it saws it in half and leaves the far half floating with nothing
         attached to it. So segments are never dropped — each one carries how
         far into the reading column it sits, and pays for it at draw time.
         Nodes, which are point marks rather than continuity, stay excluded.

         This is a ramp, not a flag, and that distinction is the whole fix. A
         boolean crossing test steps the ink by 6x between two adjacent
         segments, and the eye reads an abrupt change in density as a cut just
         as readily as it reads a gap — the line was continuous and still
         looked severed. Fading in over the approach means no two neighbouring
         segments ever differ enough to register as an edge. */
      const centreVeil = (x0, x1, y1) => {
        const enter = (y1 - readingZoneStart) / 150;
        if (enter <= 0) return 0;
        const mid = W * 0.5;
        const half = Math.max(1, mid - railInset);
        const lateral = 1 - Math.abs((x0 + x1) * 0.5 - mid) / half;
        if (lateral <= 0) return 0;
        /* sqrt pushes the veil outward: a segment only has to be *somewhat*
           inside the column to lose most of its ink, so the reading column
           stays clean instead of fading gradually from the rail inward. */
        return Math.min(1, enter) * Math.min(1, Math.sqrt(lateral));
      };

      /* Narrow pages carry extra trunks (see trunkDefs below), so they get a
         larger budget — otherwise the cap would truncate the colony long
         before it reaches the footer on a tall stacked document. */
      const segCap = W < 700 ? 13500 : 9000;
      const grow = (x, y, ang, len, depth, spread) => {
        if (segs.length > segCap) return;
        /* Once a root has cleared the fruiting body, commit it to the nearest
           outward direction. Branch noise still makes the paths organic, but
           this persistent steering prevents long trunks from wandering back
           through the page's reading column. */
        const corridorX = Math.cos(ang) < 0 ? corridorLeft : corridorRight;
        const step = 7 + rnd() * 7;
        const n = Math.max(4, Math.round(len / step));
        const pts = [[x, y]];
        let a = ang, cx = x, cy = y;
        const kids = [];
        for (let i = 0; i < n; i++) {
          a += (rnd() - 0.5) * 0.34 * spread + 0.012 * Math.sin(i * 0.7 + depth);
          if (Math.sin(ang) > 0.15) a += (Math.PI * 0.5 - a) * 0.02;
          if (cy > corridorStart) {
            const laneAngle = Math.atan2(160, corridorX - cx);
            a += (laneAngle - a) * 0.065;
          }
          cx += Math.cos(a) * step;
          cy += Math.sin(a) * step;
          if (cx < 16) { cx = 16; a = Math.PI - a; }
          if (cx > W - 16) { cx = W - 16; a = Math.PI - a; }
          if (cy > H + 60 || cy < -60) break;
          pts.push([cx, cy]);
          const order = Math.min(0.995, 0.055 + 0.9 * (cy / H) + depth * 0.012 + rnd() * 0.008);
          const x0 = pts[pts.length - 2][0];
          const y0 = pts[pts.length - 2][1];
          /* Dropping a segment does not hide a filament, it cuts one in half
             and leaves the far side floating as a detached stub — which is the
             break visible beside the story column. So nothing is discarded;
             crossings are marked and veiled at draw time instead. */
          segs.push({ x0, y0, x1: cx, y1: cy, d: depth, o: order, c: centreVeil(x0, cx, cy) });
          const branchChance = (depth < 2 ? 0.032 : 0.02) * density;
          if (i > 1 && i < n - 1 && rnd() < branchChance) {
            const side = rnd() < 0.5 ? -1 : 1;
            kids.push([cx, cy, a + side * (0.5 + rnd() * 0.75), len * (0.36 + rnd() * 0.24), depth + 1, spread * 1.15, order]);
            if (cy <= readingZoneStart || cx <= railInset || cx >= W - railInset) {
              nodes.push({ x: cx, y: cy, o: order, r: 1.6 + rnd() * 2.2, d: depth });
            }
          }
        }
        if (depth <= 1 && pts.length > 6) trunks.push(pts);
        if (depth < 4) kids.forEach((k) => grow(k[0], k[1], k[2], k[3], k[4], k[5]));
      };

      const trunkDefs = [
        [ox, oy, Math.PI * 0.5, 150],
        [ox, oy, Math.PI * 0.78, H * 0.9],
        [ox, oy, Math.PI * 0.88, H * 0.72],
        [ox, oy, Math.PI * 0.22, H * 0.9],
        [ox, oy, Math.PI * 0.12, H * 0.72]
      ];
      /* On a narrow page five trunks are not enough: branches accumulate per
         pixel travelled, so the first viewport and a half below the origin
         reads as empty and the colony only becomes visible mid-page. Extra
         trunks on small widths put the network on screen right after the
         hero, the way the desktop already does. They grow interleaved rather
         than appended: growth is depth-first and capped, so latecomers would
         starve before drawing a single segment. */
      if (W < 700) {
        trunkDefs.splice(
          1, 0,
          [ox, oy, Math.PI * 0.35, H * 0.6],
          [ox, oy, Math.PI * 0.65, H * 0.6],
          [ox, oy, Math.PI * 0.5, H * 0.55]
        );
      }
      trunkDefs.forEach((t) => grow(t[0], t[1], t[2], t[3], 0, 1));

      segs.sort((a, b) => a.o - b.o);
      nodes.sort((a, b) => a.o - b.o);
      this.segs = segs; this.nodes = nodes; this.trunks = trunks;
      this._nodeCursor = 0;
      this.pulses = trunks.filter((_, i) => i > 0).slice(0, 4).map((p, i) => ({ pts: p, t: rnd(), sp: 0.045 + rnd() * 0.05, ph: i }));
    }

    drawTo(p) {
      const ctx = this.gctx;
      if (!ctx || !this.segs) return;
      /*
       * mycellios: the reference is additive light on a black page. Over the
       * ivory and paper bands that is invisible, so `dim` flips the field to
       * normal compositing and raises the ink — the same filament reads as a
       * darkening of the stock instead of a glow on top of it. `lighter` is
       * still used over the dark bands, where it is correct and where the
       * glints depend on it.
       */
      const dim = this.hasAttribute('dim');
      ctx.globalCompositeOperation = dim ? 'source-over' : 'lighter';
      const ink = dim ? 1.08 : 0.88;
      let drawn = 0;
      while (this._cursor < this.segs.length && this.segs[this._cursor].o <= p && drawn < 900) {
        const s = this.segs[this._cursor++];
        const fade = 1 / (1 + s.d * 0.55);
        /* `s.c` is how deep this segment sits in the reading column, 0..1.
           Interpolating rather than switching is what keeps the line reading as
           one filament. The floors are deliberately near zero: the colony lives
           in the page margins, and a trunk crossing the copy at a third of its
           ink read as lines drawn across the whole page. What remains at full
           depth is a ghost that keeps the path technically unbroken without
           registering as a line. */
        const floor = s.d === 0 ? 0.08 : 0.03;
        const veil = 1 - s.c * (1 - floor);
        ctx.strokeStyle = this.rgba((0.055 + 0.075 * fade) * ink * veil);
        ctx.lineWidth = Math.max(0.35, (1.45 - s.d * 0.28) * (1 - s.c * 0.45));
        ctx.beginPath();
        ctx.moveTo(s.x0, s.y0);
        ctx.lineTo(s.x1, s.y1);
        ctx.stroke();
        // The trunk's soft halo fades on the same ramp. Dropping it abruptly
        // would reintroduce exactly the density edge the ramp above removes.
        if (s.d === 0 && s.c < 1) {
          ctx.strokeStyle = this.rgba(0.022 * ink * (1 - s.c));
          ctx.lineWidth = 4.5;
          ctx.stroke();
        }
        drawn++;
      }
      while (this._nodeCursor < this.nodes.length && this.nodes[this._nodeCursor].o <= p) {
        const nd = this.nodes[this._nodeCursor++];
        const s = nd.r * 7;
        // The node sprite has a near-white core, which on paper would punch a
        // pale hole rather than mark a fork. Halved over light bands.
        ctx.globalAlpha = (dim ? 0.075 : 0.16) / (1 + nd.d * 0.5);
        ctx.drawImage(this.sprite, nd.x - s / 2, nd.y - s / 2, s, s);
        ctx.globalAlpha = 1;
      }
      ctx.globalCompositeOperation = 'source-over';
      return this._cursor >= this.segs.length || this.segs[this._cursor].o > p;
    }

    tick = () => {
      this._raf = requestAnimationFrame(this.tick);
      if (!this._originLocked && (this._frame = (this._frame || 0) + 1) % 20 === 0) {
        if (this.lockOrigin() || ++this._originTries > 12) this._originLocked = true;
      }
      if (!this.hasAttribute('manual')) this._prog = this.selfProgress();
      const target = this._prog;
      if (target > this._shown) this._shown += Math.min(0.02, (target - this._shown) * 0.09 + 0.0006);
      this.drawTo(this._shown);

      this.placeGrowthCanvas();

      const lctx = this.lctx;
      if (!lctx || this.motion === 'off' || this.reduce) return;
      if (this.storyIsPinned()) return;
      lctx.clearRect(0, 0, this.vw, this.vh);
      /* Not the host's own box: the growth canvas is moved three different
         ways across the page, and everything drawn here has to ride the
         filaments rather than the element that contains them. */
      const top = this.canvasTop();
      /* Use wall time so spores and pulses keep the same restrained speed on
         high-refresh displays and under transient frame pressure. */
      const frameNow = performance.now();
      const dt = Math.min(0.05, Math.max(0, (frameNow - (this._lastFrame || frameNow)) / 1000));
      this._lastFrame = frameNow;
      lctx.globalCompositeOperation = 'source-over';
      const now = frameNow * 0.00012;
      const hero = this._hero || (this._hero = document.querySelector('.rb-hero'));
      const heroRect = hero && hero.getBoundingClientRect();
      const sporeTop = heroRect ? Math.max(0, heroRect.top) : 0;
      const sporeBottom = heroRect ? Math.min(this.vh, heroRect.bottom) : 0;
      if (sporeBottom > sporeTop) {
        /* Ambient spores belong to the opening scene. Clip the fixed canvas to
           the visible slice of the hero so they never leak into later bands. */
        lctx.save();
        lctx.beginPath();
        lctx.rect(0, sporeTop, this.vw, sporeBottom - sporeTop);
        lctx.clip();
        for (const sp of this.spores || []) {
          sp.y -= dt * sp.speed;
          if (sp.y < -0.08) sp.y = 1.08;
          const x = (sp.x + Math.sin(now + sp.phase) * sp.drift) * this.vw;
          const y = sp.y * this.vh;
          const edgeFade = Math.sin(Math.PI * Math.max(0, Math.min(1, (sp.y + 0.04) / 1.08)));
          lctx.globalAlpha = sp.alpha * edgeFade;
          lctx.drawImage(this.sporeSprite, x - sp.size / 2, y - sp.size / 2, sp.size, sp.size);
        }
        lctx.restore();
      }
      if (this.pulses && this._shown >= 0.02) {
        lctx.globalCompositeOperation = 'lighter';
        for (const pu of this.pulses) {
          pu.t += dt * pu.sp;
          if (pu.t > 1) pu.t -= 1;
          const idx = pu.t * (pu.pts.length - 1);
          const i0 = Math.floor(idx), f = idx - i0;
          const a = pu.pts[i0], b = pu.pts[Math.min(pu.pts.length - 1, i0 + 1)];
          const x = a[0] + (b[0] - a[0]) * f;
          const y = a[1] + (b[1] - a[1]) * f + top;
          if (y < -40 || y > this.vh + 40) continue;
          const documentY = y - top;
          const railInset = this.vw * 0.32;
          const readingZoneStart = (this._oy || 0) + Math.min(140, Math.max(96, this.vh * 0.14));
          if (documentY > readingZoneStart && x > railInset && x < this.vw - railInset) continue;
          const orderHere = Math.min(0.995, 0.055 + 0.9 * ((y - top) / this._h));
          if (orderHere > this._shown) continue;
          const s = 22 + Math.sin(pu.t * 20 + pu.ph) * 4;
          lctx.globalAlpha = 0.28 * Math.sin(Math.PI * Math.min(1, pu.t * 1.6));
          lctx.drawImage(this.sprite, x - s / 2, y - s / 2, s, s);
        }
      }
      lctx.globalAlpha = 1;
      lctx.globalCompositeOperation = 'source-over';
    };
  }
  customElements.define('mycelium-field', MyceliumField);
}
