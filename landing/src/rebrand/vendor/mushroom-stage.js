/* <mushroom-stage> — procedural translucent mushroom, three.js + custom bloom.
   Attributes: accent (#hex) | spores ("on"/"off") | motion ("full"|"calm"|"off") | scale (number)
               | variant ("hero"|"gutter"|"colony", boot-time only) */
/*
 * Vendored from the design reference. It is kept as a custom element rather
 * than rewritten as a React component on purpose: the value here is the *look*,
 * and the look lives in a hundred tuned constants (profile tables, sheen, bloom
 * thresholds) that a rewrite would quietly drift away from. React owns the
 * mount point; this owns the pixels.
 *
 * Changed against the reference in exactly six places, all marked `mycellios:`
 *   1. three.js is resolved from the bundle instead of fetched from unpkg;
 *   2. the IIFE became an idempotent exported `defineMushroomStage()`, so
 *      React's StrictMode double-mount cannot race `customElements.define`;
 *   3. a `mushroom-ready` event fires on the first *drawn* frame, so the host
 *      can fade the organism in rather than letting it pop;
 *   4. three.js is imported by name rather than as a namespace, so the bundler
 *      can drop what this file never touches — that alone is 190KB gzipped
 *      down to 137KB. `THREE_MODULE` is rebuilt from those bindings so the
 *      several hundred `THREE.Foo` call sites below stay exactly as written;
 *   5. the tuned constants that describe *which* mushroom this is were lifted
 *      into a `VARIANTS` table so the page can grow a second, younger specimen
 *      from the same renderer. The hero's entry holds the reference's own
 *      numbers verbatim and is pinned by a test;
 *   6. one body per element became N: `buildBody()` is the reference's body
 *      code lifted whole, and a variant may carry a `colony` of them. A
 *      variant without one is a single body at the identity transform, which
 *      is what the reference always built.
 *
 * Everything else — the scroll-driven `progress`, the bloom chain — is
 * untouched.
 */
import {AdditiveBlending, BufferGeometry, CanvasTexture, Clock, Color, DirectionalLight, DoubleSide, Float32BufferAttribute, FrontSide, Group, HalfFloatType, HemisphereLight, InstancedMesh, LatheGeometry, LinearFilter, Matrix4, Mesh, MeshBasicMaterial, MeshPhysicalMaterial, MeshStandardMaterial, OrthographicCamera, PerspectiveCamera, PlaneGeometry, PointLight, Points, SRGBColorSpace, Scene, ShaderMaterial, Vector2, Vector3, WebGLRenderTarget, WebGLRenderer} from "three";
const THREE_MODULE = {AdditiveBlending, BufferGeometry, CanvasTexture, Clock, Color, DirectionalLight, DoubleSide, Float32BufferAttribute, FrontSide, Group, HalfFloatType, HemisphereLight, InstancedMesh, LatheGeometry, LinearFilter, Matrix4, Mesh, MeshBasicMaterial, MeshPhysicalMaterial, MeshStandardMaterial, OrthographicCamera, PerspectiveCamera, PlaneGeometry, PointLight, Points, SRGBColorSpace, Scene, ShaderMaterial, Vector2, Vector3, WebGLRenderTarget, WebGLRenderer};


// cap silhouette: apex -> rim
const CAP_TOP = [
  [0.000, 0.955], [0.090, 0.951], [0.190, 0.938], [0.300, 0.914], [0.410, 0.878],
  [0.520, 0.830], [0.630, 0.770], [0.730, 0.700], [0.820, 0.620], [0.895, 0.535],
  [0.950, 0.452], [0.990, 0.378], [1.030, 0.318], [1.060, 0.292], [1.080, 0.302]
];
// cap underside: stem junction -> rim lip
const CAP_UNDER = [
  [0.095, 0.520], [0.220, 0.498], [0.360, 0.462], [0.500, 0.420], [0.650, 0.371],
  [0.790, 0.325], [0.910, 0.295], [1.010, 0.284], [1.060, 0.290], [1.080, 0.302]
];
const STEM = [
  [0.000, 0.560], [0.078, 0.548], [0.070, 0.400], [0.072, 0.200], [0.080, -0.040],
  [0.093, -0.300], [0.112, -0.560], [0.140, -0.820], [0.170, -1.080], [0.000, -1.180]
];

/*
 * mycellios: a fifth change against the reference — a second specimen.
 *
 * The page needs another fruiting body beside the pricing rail, and the only
 * two honest options were a flat SVG or this. A drawn one loses: the hero has
 * already shown the visitor what this organism looks like with volume on it,
 * so a flat one further down does not read as a quieter version of the same
 * thing, it reads as a worse one. And on desktop the renderer's cost is
 * already paid — three.js is in the bundle by the time anyone scrolls this
 * far, so a second instance costs a context and a few hundred triangles, not
 * another 133KB.
 *
 * But it must not be the same mushroom twice. Two identical organisms on one
 * page is wallpaper; what makes a colony read as alive is that its bodies are
 * at different ages. So `hero` is a mature open cap and `gutter` is a young
 * closed bell on a long thin stem — the stage before the cap opens, which is
 * a genuinely different silhouette rather than the same one scaled down.
 *
 * EVERY NUMBER UNDER `hero` IS THE REFERENCE'S OWN VALUE, MOVED, NOT CHANGED.
 * That is the whole safety property of this table: parameterising a look that
 * lives in tuned constants is only safe if the original constants survive
 * verbatim, so the hero is pinned against these literals by a test and cannot
 * drift while the second specimen is tuned.
 */
export const VARIANTS = {
  hero: {
    capTop: CAP_TOP,
    capUnder: CAP_UNDER,
    stem: STEM,
    capScale: [1.1, 0.86, 1.1],
    stemBend: -0.075,
    gillFactor: 1,
    camera: { x: 0.30, y: -1.02, z: 5.15, fov: 26, target: [0, 0.36, 0] },
    /* Multipliers on everything that emits light. The hero is the page's
       light source; the gutter body is lit by it and gives nothing back. */
    tune: { halo: 1, glow: 1, shell: 1, bloom: 0.68, threshold: 0.42 }
  },
  /*
   * A young bell. The cap has not opened, so it is taller than it is wide —
   * 1.29 of height against 0.65 of radius, where the hero's is the other way
   * round — and it sits on a stem half the thickness. Read as a silhouette,
   * which is how it will mostly be read at 110px in a margin, it is a
   * different organism at a glance and unmistakably the same species.
   */
  gutter: {
    capTop: [
      [0.000, 1.215], [0.055, 1.208], [0.115, 1.190], [0.180, 1.158], [0.250, 1.110],
      [0.320, 1.048], [0.390, 0.972], [0.452, 0.888], [0.508, 0.796], [0.555, 0.698],
      [0.592, 0.598], [0.618, 0.500], [0.634, 0.412], [0.643, 0.348], [0.648, 0.318]
    ],
    capUnder: [
      [0.058, 0.560], [0.140, 0.548], [0.230, 0.522], [0.320, 0.484], [0.410, 0.436],
      [0.490, 0.392], [0.560, 0.352], [0.612, 0.328], [0.640, 0.319], [0.648, 0.318]
    ],
    stem: [
      [0.000, 0.560], [0.046, 0.552], [0.041, 0.400], [0.040, 0.180], [0.043, -0.080],
      [0.050, -0.340], [0.060, -0.620], [0.074, -0.900], [0.092, -1.160], [0.000, -1.240]
    ],
    /* Unsquashed: the hero's 0.86 flattens a dome that has already opened.
       Doing the same to a bell would just make it a dome again. */
    capScale: [1, 1, 1],
    /* More lean than the hero's. A thin stem that stands plumb is a pin; the
       bend is most of what says this grew rather than was placed. */
    stemBend: -0.105,
    /* A closed bell shows almost none of its gills, and the ones it does show
       are a dark seam under the rim rather than a fan. Half the blades at a
       sixth of the emission is what that looks like from outside. */
    gillFactor: 0.5,
    /*
     * Solved against the geometry rather than eyeballed, because this framing
     * carries a load-bearing claim: the body has to be *cut* by the section's
     * bottom rule, not stood on it.
     *
     * Projecting the profile through this camera at the wrapper's 0.62 aspect
     * puts the crown at 0.053 down the box and the foot of the stem at 1.061 —
     * just past the bottom edge, which the wrapper clips. So the stem runs into
     * the rule and disappears under it, which is what something growing out of
     * a surface does; a stem that stopped at 0.99 would be an object resting on
     * a line. Sideways it spans 0.07..0.93, so the cap has margin at the widest
     * point of its sway. The check is in the tests, run against these numbers.
     */
    camera: { x: 0.16, y: -1.62, z: 5.10, fov: 25, target: [0, 0.06, 0] },
    /* The subject of this section is the pricing rail. Killing the halo
       outright and dropping the rest to a third is what keeps this texture in
       a margin instead of a second thing to look at — low contrast, which is
       what actually prevents competition, rather than no volume. */
    tune: { halo: 0, glow: 0.34, shell: 0.42, bloom: 0.26, threshold: 0.6 }
  },
  /*
   * Three bodies, one context, one patch of ground.
   *
   * The questions column is a tall empty space under a sticky heading, and one
   * mushroom in it would be a spot of decoration. Three at different ages are a
   * colony, which is the thing the company is actually named after — and a
   * colony is the one arrangement that earns the space, because the reason to
   * look at it is the relationship between the bodies rather than any one of
   * them.
   *
   * The ages are the point. `flared` is past its prime, cap turned up at the
   * brim and gone concave; `opening` is the bell in the middle of tearing open;
   * `button` is a knob on a thick short stem that has barely cleared the
   * ground. Together with the hero's mature plate and the gutter's young bell
   * that is five distinct silhouettes of one species, and none of them is
   * another one scaled.
   *
   * They are placed on a shallow arc rather than a line — different z, so the
   * front one overlaps the back one and the group has depth from a single
   * camera. `rotY` turns each one a different way so no two present the same
   * profile.
   */
  colony: {
    /* The colony's own frame is the whole group, so the top-level profile keys
       are the middle specimen's: anything that reads `V.capTop` without knowing
       about colonies (a probe, a future caller) gets a real body rather than
       undefined. */
    capTop: [
      [0.000, 1.015], [0.241, 1.008], [0.363, 0.986], [0.460, 0.952], [0.540, 0.905],
      [0.607, 0.847], [0.664, 0.781], [0.712, 0.710], [0.750, 0.635], [0.780, 0.561],
      [0.801, 0.491], [0.814, 0.430], [0.818, 0.390], [0.850, 0.390], [0.870, 0.430]
    ],
    capUnder: [
      [0.080, 0.645], [0.168, 0.609], [0.256, 0.576], [0.343, 0.545], [0.431, 0.516],
      [0.519, 0.491], [0.607, 0.469], [0.694, 0.451], [0.782, 0.437], [0.870, 0.430]
    ],
    stem: [
      [0.000, 0.645], [0.064, 0.645], [0.059, 0.401], [0.059, 0.083], [0.071, -0.292],
      [0.080, -0.667], [0.087, -0.968], [0.092, -1.155], [0.094, -1.230], [0.000, -1.290]
    ],
    capScale: [1, 1, 1],
    stemBend: -0.085,
    gillFactor: 0.8,
    colony: [
      {
        /* Old. The cap has gone past flat: the brim sits 0.105 above the low
           point of the margin, so the underside is convex and catches light
           from below instead of shading itself. That upturn is the whole
           reason this one is legible as elderly at 200px. */
        capTop: [
          [0.000, 0.735], [0.425, 0.730], [0.558, 0.717], [0.653, 0.696], [0.727, 0.669],
          [0.786, 0.637], [0.835, 0.603], [0.874, 0.568], [0.905, 0.536], [0.929, 0.509],
          [0.946, 0.488], [0.956, 0.475], [0.959, 0.470], [0.997, 0.470], [1.020, 0.575]
        ],
        capUnder: [
          [0.070, 0.452], [0.176, 0.473], [0.281, 0.492], [0.387, 0.509], [0.492, 0.526],
          [0.598, 0.540], [0.703, 0.553], [0.809, 0.563], [0.914, 0.571], [1.020, 0.575]
        ],
        /* The last three rows carry the stem to the common ground depth. See the
           note on GROUND below: the profile's own foot is not a design choice,
           it is whatever puts this body's tip under the same plane as the other
           two once `scale` and `at[1]` have been applied. */
        stem: [
          [0.000, 0.452], [0.052, 0.452], [0.048, 0.240], [0.048, -0.038], [0.060, -0.364],
          [0.071, -0.690], [0.078, -0.952], [0.084, -1.115], [0.086, -1.180], [0.088, -1.318],
          [0.090, -1.457], [0.000, -1.538]
        ],
        /* A cap this old has thinned as well as spread. */
        capScale: [1, 0.94, 1],
        stemBend: -0.055,
        gillFactor: 1,
        at: [-0.98, -0.02, -0.30], scale: 0.92, rotY: 0.42
      },
      {
        /* Middle. Still a bell, but the veil has torn and the margin is on its
           way down and out — the only one of the three showing the transition
           rather than an end state. */
        capTop: [
          [0.000, 1.015], [0.241, 1.008], [0.363, 0.986], [0.460, 0.952], [0.540, 0.905],
          [0.607, 0.847], [0.664, 0.781], [0.712, 0.710], [0.750, 0.635], [0.780, 0.561],
          [0.801, 0.491], [0.814, 0.430], [0.818, 0.390], [0.850, 0.390], [0.870, 0.430]
        ],
        capUnder: [
          [0.080, 0.645], [0.168, 0.609], [0.256, 0.576], [0.343, 0.545], [0.431, 0.516],
          [0.519, 0.491], [0.607, 0.469], [0.694, 0.451], [0.782, 0.437], [0.870, 0.430]
        ],
        stem: [
          [0.000, 0.645], [0.064, 0.645], [0.059, 0.401], [0.059, 0.083], [0.071, -0.292],
          [0.080, -0.667], [0.087, -0.968], [0.092, -1.155], [0.094, -1.230], [0.095, -1.305],
          [0.096, -1.380], [0.000, -1.455]
        ],
        capScale: [1, 1, 1],
        stemBend: -0.085,
        gillFactor: 0.8,
        at: [0.66, 0.02, 0.08], scale: 1, rotY: -0.58
      },
      {
        /* Youngest, and the one that makes the group a colony rather than a
           row: it is small, in front, and leaning hard. Its stem is 0.118 at
           the top against the old one's 0.052 — a button's stem is thick
           relative to its cap, and getting that ratio right is what stops it
           reading as the middle one shrunk. */
        capTop: [
          [0.000, 1.045], [0.081, 1.042], [0.141, 1.031], [0.195, 1.014], [0.242, 0.992],
          [0.285, 0.964], [0.322, 0.933], [0.354, 0.898], [0.380, 0.862], [0.401, 0.826],
          [0.416, 0.793], [0.425, 0.763], [0.428, 0.744], [0.445, 0.744], [0.455, 0.760]
        ],
        capUnder: [
          [0.105, 0.885], [0.144, 0.864], [0.183, 0.845], [0.222, 0.827], [0.261, 0.810],
          [0.299, 0.796], [0.338, 0.783], [0.377, 0.772], [0.416, 0.764], [0.455, 0.760]
        ],
        /* The deepest profile of the three, which is the opposite of what a
           button's stem looks like — because this body is scaled 0.66 and sits
           0.34 low, so it needs 0.65 more local stem than the middle one to
           reach the same world ground. The visible stem is still the shortest of
           the three; the surplus is below the edge where nobody sees it. */
        stem: [
          [0.000, 0.885], [0.118, 0.885], [0.111, 0.653], [0.111, 0.350], [0.125, -0.008],
          [0.137, -0.365], [0.144, -0.650], [0.150, -0.829], [0.152, -0.900], [0.154, -1.029],
          [0.156, -1.158], [0.157, -1.287], [0.159, -1.416], [0.161, -1.545], [0.000, -1.659]
        ],
        capScale: [1, 1, 1],
        /* The most lean of the three. A button is the least anchored thing in
           the patch. */
        stemBend: -0.13,
        /* A closed knob shows essentially no gills. */
        gillFactor: 0.45,
        at: [-0.12, -0.34, 0.62], scale: 0.66, rotY: 1.15
      }
    ],
    /*
     * Solved against all three bodies at once, the same way the gutter's was,
     * and over the whole rotation the loop sways the group through — 0.01 to
     * 0.57 radians — rather than at rest, so the framing holds at the extremes
     * of the sway instead of only in the middle of it.
     *
     * At the wrapper's 1.62 aspect the colony spans 0.021..0.973 across the box
     * with 0.006 of skew, so it is centred and fills the width; crowns sit
     * 0.073 down from the top.
     *
     * GROUND. The three stems end on one world plane at y = -1.36, and every
     * tip projects below 1.10 — past the bottom edge, which the wrapper clips.
     * That plane is the fix for the first version, where each profile simply
     * kept its own foot depth: after `scale` and `at[1]` the button's tip landed
     * at 0.989 of the box at rest and 0.911 when scrolled, so its closing cone
     * hung *above* the edge with shading on it, and the group read as three
     * mushrooms floating rather than three growing out of something. The other
     * two reached only 1.004 and 1.076 — resting on the line, which is the same
     * mistake with less of it.
     *
     * So the depth of each profile's last rows is not a drawing decision. It is
     * solved: whatever local y puts this body's tip under y = -1.36 in world
     * space once its own scale and offset apply, plus enough margin that the
     * scroll lift (0.14) and the breath (0.0125) cannot raise it back into view.
     * Per-body, because a single worst-case over the group is exactly what let
     * one floating specimen through the first time.
     */
    camera: { x: 0, y: -2.04, z: 5.10, fov: 24, target: [-0.2, 0.2, 0] },
    /* Lower than the gutter's even. This sits beside six rows of text that a
       visitor is reading, and the halo is the one part that would put light
       across the words. */
    tune: { halo: 0, glow: 0.3, shell: 0.36, bloom: 0.22, threshold: 0.62 }
  }
};

export function defineMushroomStage() {
  if (window.customElements && customElements.get('mushroom-stage')) return;
  /* mycellios: the reference did `import('https://unpkg.com/three@0.184.0/...')`
     at runtime. A production landing must not fetch its renderer from a third
     party CDN on first paint — it is an availability dependency and a privacy
     leak. Resolved from the bundle, kept async so the ~150KB of three still
     lands in its own chunk and never blocks the hero's text. */
  const loadThree = () => Promise.resolve(THREE_MODULE);

  const QUAD_VS = `varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

  const BRIGHT_FS = `
    uniform sampler2D tDiffuse; uniform float uThreshold, uKnee; varying vec2 vUv;
    void main(){
      vec4 c = texture2D(tDiffuse, vUv);
      float l = max(c.r, max(c.g, c.b));
      float w = smoothstep(uThreshold, uThreshold + uKnee, l);
      gl_FragColor = vec4(c.rgb * w, 1.0);
    }`;

  const BLUR_FS = `
    uniform sampler2D tDiffuse; uniform vec2 uDir; varying vec2 vUv;
    void main(){
      vec3 s = texture2D(tDiffuse, vUv).rgb * 0.227027;
      s += texture2D(tDiffuse, vUv + uDir * 1.3846154).rgb * 0.3162162;
      s += texture2D(tDiffuse, vUv - uDir * 1.3846154).rgb * 0.3162162;
      s += texture2D(tDiffuse, vUv + uDir * 3.2307692).rgb * 0.0702703;
      s += texture2D(tDiffuse, vUv - uDir * 3.2307692).rgb * 0.0702703;
      gl_FragColor = vec4(s, 1.0);
    }`;

  const COMP_FS = `
    uniform sampler2D tScene, tBloom; uniform float uStrength; varying vec2 vUv;
    vec3 aces(vec3 x){ return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
    void main(){
      vec4 s = texture2D(tScene, vUv);
      vec3 b = texture2D(tBloom, vUv).rgb * uStrength;
      vec3 c = pow(aces(s.rgb + b), vec3(0.4545455));
      float ba = clamp(max(b.r, max(b.g, b.b)) * 1.35, 0.0, 1.0);
      gl_FragColor = vec4(c, clamp(s.a + ba, 0.0, 1.0));
    }`;

  const lerpTable = (tbl, x) => {
    if (x <= tbl[0][0]) return tbl[0][1];
    for (let i = 1; i < tbl.length; i++) {
      if (x <= tbl[i][0]) {
        const [x0, y0] = tbl[i - 1], [x1, y1] = tbl[i];
        return y0 + (y1 - y0) * ((x - x0) / (x1 - x0));
      }
    }
    return tbl[tbl.length - 1][1];
  };


  const radialTexture = (THREE, inner, mid) => {
    const c = document.createElement('canvas');
    c.width = c.height = 256;
    const g = c.getContext('2d');
    const grd = g.createRadialGradient(128, 128, 0, 128, 128, 128);
    grd.addColorStop(0, inner);
    grd.addColorStop(0.28, mid);
    grd.addColorStop(1, 'rgba(0,0,0,0)');
    g.fillStyle = grd;
    g.fillRect(0, 0, 256, 256);
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  };

  class MushroomStage extends HTMLElement {
    /* `variant` is deliberately absent: it chooses geometry, and geometry is
       built once at boot. Making it live would mean rebuilding and disposing
       every buffer on an attribute write, which is a lot of machinery for a
       thing no caller does. Set it before the element is connected. */
    static get observedAttributes() { return ['accent', 'spores', 'motion', 'scale', 'progress']; }

    setProgress(p) { this._prog = Math.max(0, Math.min(1, p || 0)); }

    connectedCallback() {
      this._dead = false;
      if (this._booted) {
        if (this._ready) { cancelAnimationFrame(this._raf); if (this._ro) this._ro.observe(this); if (this._io) this._io.observe(this); this.tick(); }
        return;
      }
      this._booted = true;
      this.style.cssText += ';display:block;position:absolute;inset:0;width:100%;height:100%;';
      this.boot().catch((e) => console.warn('[mushroom-stage] boot failed', e));
    }

    attributeChangedCallback(name, _o, v) {
      if (!this._ready) return;
      if (name === 'accent') this.applyAccent(v);
      if (name === 'spores' && this.spores) this.spores.visible = v !== 'off' && v !== 'false';
      if (name === 'motion') this.motion = v || 'full';
      if (name === 'scale') this.baseScale = parseFloat(v) || 1;
      if (name === 'progress') this.setProgress(parseFloat(v));
    }

    disconnectedCallback() {
      this._dead = true;
      cancelAnimationFrame(this._raf);
      if (this._ro) this._ro.disconnect();
      if (this._io) this._io.disconnect();
      window.removeEventListener('pointermove', this._onMove);
      removeEventListener('scroll', this._onScroll);
      removeEventListener('resize', this._onScroll);
      if (this.renderer) { this.renderer.dispose(); this.renderer.forceContextLoss?.(); }
    }

    async boot() {
      const THREE = await loadThree();
      if (this._dead) return;
      this.THREE = THREE;
      /* Unknown names fall back to the hero rather than throwing: a typo in an
         attribute should cost the wrong specimen, not an empty canvas where the
         organism was. */
      const V = VARIANTS[this.getAttribute('variant')] || VARIANTS.hero;
      this.V = V;
      this.motion = this.getAttribute('motion') || 'full';
      this.baseScale = parseFloat(this.getAttribute('scale')) || 1;
      const reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
      this.reduce = reduce;
      const mobileQuality = this.getAttribute('quality') === 'mobile';
      const lowPower = mobileQuality || (navigator.hardwareConcurrency || 8) <= 4 || innerWidth < 700;

      const canvas = document.createElement('canvas');
      canvas.style.cssText = 'display:block;width:100%;height:100%;';
      this.appendChild(canvas);
      this.canvas = canvas;

      const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: !lowPower, powerPreference: 'high-performance' });
      renderer.setClearColor(0x000000, 0);
      this.renderer = renderer;
      this.dprCap = mobileQuality ? 1 : lowPower ? 1.15 : 1.4;

      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(V.camera.fov, 1, 0.1, 60);
      camera.position.set(V.camera.x, V.camera.y, V.camera.z);
      this.scene = scene; this.camera = camera;
      this.target = new THREE.Vector3(...V.camera.target);

      const accent = new THREE.Color(this.getAttribute('accent') || '#C9976A');
      this.accent = accent;
      this._prog = this._prog || 0;
      this._progD = 0;
      this._flow = 0;

      // ── materials
      const skin = new THREE.MeshPhysicalMaterial({
        color: 0xded7cc, roughness: 0.42, metalness: 0, clearcoat: 0.5, clearcoatRoughness: 0.45,
        sheen: 1, sheenRoughness: 0.6, sheenColor: new THREE.Color(0xffdcb8), ior: 1.36,
        side: THREE.DoubleSide, transparent: true, opacity: 0.88
      });
      skin.name = 'skin';
      const underSkin = new THREE.MeshPhysicalMaterial({
        color: 0xe4dcd0, roughness: 0.55, metalness: 0, sheen: 0.6, sheenColor: new THREE.Color(accent),
        emissive: accent.clone(), emissiveIntensity: 0.3, side: THREE.DoubleSide
      });
      const gillMat = new THREE.MeshStandardMaterial({
        color: 0xe9e0d3, roughness: 0.45, metalness: 0, emissive: accent.clone(), emissiveIntensity: 0.32,
        side: THREE.DoubleSide, vertexColors: true, transparent: true, opacity: 0.94
      });
      this.mats = { skin, underSkin, gillMat };

      const group = new THREE.Group();
      group.name = 'mushroom';
      scene.add(group);
      this.group = group;

      /*
       * One element, one context, however many bodies.
       *
       * The questions column wanted three specimens, and the obvious way to get
       * them — three <mushroom-stage> elements — would have put five live WebGL
       * contexts on one page next to the hero and the gutter. Browsers cap
       * contexts somewhere around eight to sixteen and drop the oldest when you
       * pass it, so the obvious way buys a page where scrolling far enough kills
       * the hero. Three bodies in one scene cost three draw calls instead.
       *
       * It is also the better picture. Three separate canvases are three
       * pictures of a mushroom; three bodies in one scene share a camera, a key
       * light and a horizon, so they stand in the same place — which is what
       * makes them read as a colony rather than as repeated decoration.
       *
       * A variant without a `colony` is its own single body, placed at the
       * origin at unit scale: an identity transform, so the hero and the gutter
       * come out of this loop exactly as they went in.
       */
      const seg = lowPower ? 56 : 80;
      const members = V.colony || [V];
      this.bodies = members.map((B) => this.buildBody(THREE, B, { group, seg, lowPower, skin, underSkin, gillMat, accent, tune: V.tune, scene }));

      this.capGroup = this.bodies[0].capGroup;
      this.shellMat = this.bodies[0].shellMat;
      this.stem = this.bodies[0].stem;
      return this.finishBoot(THREE, { scene, group, V, accent, lowPower, canvas });
    }

    /* Everything that is one fruiting body: cap, shell, gills, stem, and the
       light inside it. Pulled out of boot() when the colony arrived — three
       copies of a hundred lines inline is how the specimens would have drifted
       apart. */
    buildBody(THREE, B, ctx) {
      const { group, seg, lowPower, skin, underSkin, gillMat, accent, tune, scene } = ctx;
      const V = B;

      const root = new THREE.Group();
      root.name = 'body';
      /* Placement is optional and defaults to the identity, so a lone specimen
         is not paying for the colony's existence. */
      root.position.set(...(B.at || [0, 0, 0]));
      root.scale.setScalar(B.scale || 1);
      root.rotation.y = B.rotY || 0;
      group.add(root);

      // ── cap
      const capTopGeo = new THREE.LatheGeometry(V.capTop.map(p => new THREE.Vector2(p[0], p[1])), seg);
      const cap = new THREE.Mesh(capTopGeo, skin);
      cap.name = 'cap';
      const underGeo = new THREE.LatheGeometry(V.capUnder.map(p => new THREE.Vector2(p[0], p[1])), seg);
      const under = new THREE.Mesh(underGeo, underSkin);
      under.name = 'capUnderside';
      const capGroup = new THREE.Group();
      capGroup.scale.set(...V.capScale);
      capGroup.add(cap, under);
      root.add(capGroup);

      // ── fresnel shell over the cap
      const shellMat = new THREE.ShaderMaterial({
        uniforms: { uColor: { value: new THREE.Color(0xffeedc) }, uPower: { value: 2.4 }, uStrength: { value: 0.85 * tune.shell } },
        vertexShader: `varying vec3 vN; varying vec3 vV;
          void main(){ vec4 mv = modelViewMatrix * vec4(position,1.0); vN = normalize(normalMatrix * normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix * mv; }`,
        fragmentShader: `uniform vec3 uColor; uniform float uPower, uStrength; varying vec3 vN; varying vec3 vV;
          void main(){ float f = pow(1.0 - clamp(dot(normalize(vN), normalize(vV)), 0.0, 1.0), uPower);
          gl_FragColor = vec4(uColor * f * uStrength, 1.0); }`,
        transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.FrontSide
      });
      const shell = new THREE.Mesh(capTopGeo, shellMat);
      shell.scale.setScalar(1.016);
      shell.name = 'capRim';
      capGroup.add(shell);

      // ── gills
      const bladeGeo = (r0, r1) => {
        const N = 15, pos = [], col = [], idx = [];
        for (let i = 0; i <= N; i++) {
          const t = i / N;
          const r = r0 + (r1 - r0) * t;
          const yTop = lerpTable(V.capUnder, r) - 0.005;
          const d = 0.072 * Math.pow(t, 0.45) * (1 - Math.pow(t, 5));
          pos.push(r, yTop, 0, r, yTop - d, 0);
          const c = 0.55 + 0.45 * (1 - t);
          col.push(c, c, c, c * 0.6, c * 0.6, c * 0.6);
        }
        for (let i = 0; i < N; i++) {
          const a = i * 2;
          idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
        }
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
        g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
        g.setIndex(idx);
        g.computeVertexNormals();
        return g;
      };
      /*
       * Gill radii are expressed against the rim rather than as absolutes.
       *
       * The literals below are the reference's, which reach 1.052 of a rim that
       * sits at 1.08 — so they end just inside it. Kept as absolutes they would
       * have run a long way past a narrower cap's rim and hung in the air
       * outside the body. `rim / 1.08` is exactly 1 for the hero, so its blades
       * are unchanged to the bit, and the other specimen's are re-fitted to
       * its own underside without a second table.
       */
      const rimK = V.capUnder[V.capUnder.length - 1][0] / 1.08;
      /* A closed bell hides most of its gills; the hero shows all of them. */
      const count = Math.round((lowPower ? 44 : 64) * V.gillFactor);
      const long = new THREE.InstancedMesh(bladeGeo(0.135 * rimK, 1.052 * rimK), gillMat, count);
      const short = new THREE.InstancedMesh(bladeGeo(0.48 * rimK, 1.048 * rimK), gillMat, count);
      long.name = 'gills'; short.name = 'gillsShort';
      const m = new THREE.Matrix4();
      const bladeCol = new THREE.Color();
      for (let i = 0; i < count; i++) {
        const a = (i / count) * Math.PI * 2;
        m.makeRotationY(a + (Math.random() - 0.5) * 0.012);
        long.setMatrixAt(i, m);
        m.makeRotationY(a + Math.PI / count);
        short.setMatrixAt(i, m);
        const k = i % 2 ? 0.52 : 1;
        bladeCol.setScalar(k * (0.9 + Math.random() * 0.1));
        long.setColorAt(i, bladeCol);
        bladeCol.setScalar((i % 2 ? 1 : 0.6) * 0.85);
        short.setColorAt(i, bladeCol);
      }
      long.instanceColor.needsUpdate = true;
      short.instanceColor.needsUpdate = true;
      long.instanceMatrix.needsUpdate = short.instanceMatrix.needsUpdate = true;
      capGroup.add(long, short);

      // ── stem, bent
      const stemGeo = new THREE.LatheGeometry(V.stem.map(p => new THREE.Vector2(p[0], p[1])), Math.max(40, seg / 2));
      {
        const p = stemGeo.attributes.position;
        for (let i = 0; i < p.count; i++) {
          const y = p.getY(i);
          const k = Math.max(0, 0.56 - y) / 1.74;
          const off = V.stemBend * Math.pow(k, 1.75);
          p.setX(i, p.getX(i) + off);
          p.setZ(i, p.getZ(i) + off * 0.35);
        }
        p.needsUpdate = true;
        stemGeo.computeVertexNormals();
      }
      const stem = new THREE.Mesh(stemGeo, skin);
      stem.name = 'stem';
      root.add(stem);

      /* The light that makes a body look lit from inside rather than painted.
         It belongs to the body, not the scene, so a colony member carries its
         own — three specimens sharing one point light at the origin would leave
         the outer two flat. */
      const inner = new THREE.PointLight(accent.clone(), 1.3 * tune.glow, 2.6, 2);
      inner.position.set(0, 0.44, 0);
      root.add(inner);

      return { root, capGroup, stem, shellMat, inner, capScale: V.capScale };
    }

    finishBoot(THREE, ctx) {
      const { scene, group, V, accent, lowPower, canvas } = ctx;
      const gillMat = this.mats.gillMat;

      // ── halos
      const haloBig = new THREE.Mesh(
        new THREE.PlaneGeometry(1, 1),
        new THREE.MeshBasicMaterial({
          map: radialTexture(THREE, 'rgba(255,246,235,0.5)', 'rgba(201,151,106,0.17)'),
          transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false, opacity: 0.5
        })
      );
      haloBig.name = 'halo';
      haloBig.position.set(0, 0.36, -1.1);
      haloBig.scale.setScalar(4.2);
      haloBig.renderOrder = -2;
      const haloCore = new THREE.Mesh(
        new THREE.PlaneGeometry(1, 1),
        new THREE.MeshBasicMaterial({
          map: radialTexture(THREE, 'rgba(255,238,220,0.8)', 'rgba(201,151,106,0.24)'),
          transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false, opacity: 0.42
        })
      );
      haloCore.name = 'haloCore';
      haloCore.position.set(0, 0.30, -0.35);
      haloCore.scale.setScalar(1.7);
      haloCore.renderOrder = -1;
      group.add(haloBig, haloCore);
      /* A specimen with no halo still builds them and hides them, rather than
         omitting them: the tick loop drives `this.halos[0]` every frame, and a
         variant that changes the *shape* of the object graph is a variant that
         will eventually crash the loop for one caller and not the other. */
      haloBig.visible = haloCore.visible = V.tune.halo > 0;
      this.halos = [haloBig, haloCore];

      // ── spores
      const sporeCount = lowPower ? 34 : 70;
      const sp = new Float32Array(sporeCount * 3), rnd = new Float32Array(sporeCount);
      for (let i = 0; i < sporeCount; i++) {
        const a = Math.random() * Math.PI * 2, r = 0.15 + Math.random() * 1.15;
        sp[i * 3] = Math.cos(a) * r;
        sp[i * 3 + 1] = -0.85 + Math.random() * 2.1;
        sp[i * 3 + 2] = Math.sin(a) * r * 0.7;
        rnd[i] = Math.random();
      }
      const sporeGeo = new THREE.BufferGeometry();
      sporeGeo.setAttribute('position', new THREE.Float32BufferAttribute(sp, 3));
      sporeGeo.setAttribute('aRnd', new THREE.Float32BufferAttribute(rnd, 1));
      const sporeMat = new THREE.ShaderMaterial({
        uniforms: { uTime: { value: 0 }, uFlow: { value: 0 }, uSize: { value: 1 }, uDpr: { value: 1 }, uColor: { value: new THREE.Color(accent).lerp(new THREE.Color(0xfff2e4), 0.5) } },
        vertexShader: `attribute float aRnd; uniform float uTime, uFlow, uSize, uDpr; varying float vA;
          void main(){
            vec3 p = position;
            p.y = mod(p.y + 0.95 + uFlow * (0.45 + aRnd * 0.9), 2.2) - 0.95;
            p.x += sin(uTime * 0.32 + aRnd * 21.0) * 0.05;
            p.z += cos(uTime * 0.26 + aRnd * 13.0) * 0.05;
            vec4 mv = modelViewMatrix * vec4(p, 1.0);
            gl_PointSize = uSize * uDpr * (0.5 + aRnd * 0.9) * 26.0 / max(0.001, -mv.z);
            vA = (0.18 + 0.62 * aRnd) * smoothstep(-0.95, -0.35, p.y) * (1.0 - smoothstep(0.55, 1.2, p.y));
            gl_Position = projectionMatrix * mv;
          }`,
        fragmentShader: `uniform vec3 uColor; varying float vA;
          void main(){ float a = smoothstep(0.5, 0.02, length(gl_PointCoord - 0.5)); gl_FragColor = vec4(uColor * a * vA, a * vA); }`,
        transparent: true, blending: THREE.AdditiveBlending, depthWrite: false
      });
      const spores = new THREE.Points(sporeGeo, sporeMat);
      spores.name = 'spores';
      spores.visible = (this.getAttribute('spores') || 'on') !== 'off';
      group.add(spores);
      this.spores = spores; this.sporeMat = sporeMat;

      // ── lights
      const hemi = new THREE.HemisphereLight(0xf2e8dc, 0x0a0908, 0.5);
      const key = new THREE.DirectionalLight(0xfff6ec, 1.5); key.position.set(-2.3, 2.7, 1.9);
      const rim = new THREE.DirectionalLight(0xffdcb4, 1.35); rim.position.set(2.5, 0.7, -2.1);
      const fill = new THREE.DirectionalLight(0xb9a894, 0.5); fill.position.set(1.4, -1.4, 2.2);
      scene.add(hemi, key, rim, fill);
      /* Kept as an alias so applyAccent and the tick loop still have the single
         name they were written against; the colony drives every body's light. */
      this.inner = this.bodies[0].inner;

      // ── bloom plumbing
      const half = () => ({ minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, type: THREE.HalfFloatType, depthBuffer: false });
      this.rtScene = new THREE.WebGLRenderTarget(2, 2, { type: THREE.HalfFloatType, samples: lowPower ? 0 : 2 });
      this.rtA = new THREE.WebGLRenderTarget(2, 2, half());
      this.rtB = new THREE.WebGLRenderTarget(2, 2, half());
      const mk = (fs, uniforms) => new THREE.ShaderMaterial({ uniforms, vertexShader: QUAD_VS, fragmentShader: fs, depthTest: false, depthWrite: false });
      this.pBright = mk(BRIGHT_FS, { tDiffuse: { value: null }, uThreshold: { value: V.tune.threshold }, uKnee: { value: 0.35 } });
      this.pBlur = mk(BLUR_FS, { tDiffuse: { value: null }, uDir: { value: new THREE.Vector2() } });
      this.pComp = mk(COMP_FS, { tScene: { value: null }, tBloom: { value: null }, uStrength: { value: V.tune.bloom } });
      this.pComp.transparent = true;
      this.quadScene = new THREE.Scene();
      this.quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
      this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.pBright);
      this.quadScene.add(this.quad);

      this.clock = new THREE.Clock();
      this.pt = { x: 0, y: 0, tx: 0, ty: 0 };
      this._onMove = (e) => {
        if (this.motion === 'off' || this.reduce) return;
        const r = this.getBoundingClientRect();
        const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
        this.pt.tx = Math.max(-1, Math.min(1, (e.clientX - cx) / (innerWidth * 0.55)));
        this.pt.ty = Math.max(-1, Math.min(1, (e.clientY - cy) / (innerHeight * 0.6)));
      };
      window.addEventListener('pointermove', this._onMove, { passive: true });
      this._onScroll = () => {
        if (this.hasAttribute('manual')) return;
        const vh = innerHeight;
        const r = this.getBoundingClientRect();
        const past = Math.max(0, -r.top + vh * 0.12);
        this.setProgress(Math.min(1, past / (vh * 0.85)));
      };
      addEventListener('scroll', this._onScroll, { passive: true });
      addEventListener('resize', this._onScroll);
      this._onScroll();

      this._ro = new ResizeObserver(() => this.resize());
      this._ro.observe(this);
      this._visible = true;
      this._io = new IntersectionObserver((es) => { this._visible = es[0].isIntersecting; }, { threshold: 0 });
      this._io.observe(this);

      this._ready = true;
      this.resize();
      this.tick();
    }

    applyAccent(hex) {
      const { THREE } = this;
      if (!THREE || !hex) return;
      const c = new THREE.Color(hex);
      this.accent.copy(c);
      this.mats.gillMat.emissive.copy(c);
      this.mats.underSkin.emissive.copy(c);
      this.mats.underSkin.sheenColor.copy(c);
      this.bodies.forEach((b) => b.inner.color.copy(c));
      this.sporeMat.uniforms.uColor.value.copy(c).lerp(new THREE.Color(0xfff2e4), 0.5);
    }

    resize() {
      if (!this._ready) return;
      const r = this.getBoundingClientRect();
      const w = Math.max(2, Math.round(r.width)), h = Math.max(2, Math.round(r.height));
      const dpr = Math.min(window.devicePixelRatio || 1, this.dprCap);
      this.dpr = dpr;
      this.renderer.setPixelRatio(dpr);
      this.renderer.setSize(w, h, false);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      const pw = Math.round(w * dpr), ph = Math.round(h * dpr);
      this.rtScene.setSize(pw, ph);
      this.rtA.setSize(Math.max(2, pw >> 1), Math.max(2, ph >> 1));
      this.rtB.setSize(Math.max(2, pw >> 1), Math.max(2, ph >> 1));
      this.sporeMat.uniforms.uDpr.value = dpr;
      this._needsRender = true;
    }

    pass(mat, target) {
      this.quad.material = mat;
      this.renderer.setRenderTarget(target || null);
      this.renderer.render(this.quadScene, this.quadCam);
    }

    _dt() {
      const n = performance.now();
      const d = this._last ? (n - this._last) / 1000 : 0.016;
      this._last = n;
      return d;
    }

    tick = () => {
      this._raf = requestAnimationFrame(this.tick);
      if (!this._visible) return;
      const t = this.clock.getElapsedTime();
      const dt = Math.min(0.05, this.clock.getDelta ? this._dt() : 0.016);
      const still = this.reduce || this.motion === 'off';
      const amp = this.motion === 'calm' ? 0.5 : 1;
      this._progD += ((this._prog || 0) - this._progD) * 0.07;
      const g = this._progD;
      this._flow += dt * (0.03 + g * 0.14);
      this.sporeMat.uniforms.uFlow.value = this._flow;

      /* The loop rewrites cap scale, halo, glow and shell every frame, so the
         boot-time values are only defaults — anything tuned per specimen has to
         be re-applied here too or the first animated frame puts it straight
         back to the hero's. */
      const V = this.V;
      if (!still) {
        const p = this.pt;
        p.x += (p.tx - p.x) * 0.045;
        p.y += (p.ty - p.y) * 0.045;
        this.group.rotation.y = 0.18 + p.x * 0.17 * amp + g * 0.22;
        const breath = Math.sin(t * 0.62);
        this.group.rotation.z = -0.03 + Math.sin(t * 0.28) * 0.025 * amp - g * 0.05;
        this.group.position.y = g * 0.14 + breath * 0.025 * amp;
        this.group.rotation.x = 0.17 - p.y * 0.05 * amp + Math.sin(t * 0.22) * 0.015 * amp;
        this.group.scale.setScalar(this.baseScale * (1 + breath * 0.012 * amp));
        /* Each body breathes on its own clock. Three specimens swelling in
           lockstep would beat like one object cut into three pieces; the phase
           offset is what makes them separate organisms that happen to share a
           patch of ground. A lone specimen gets offset 0 — the hero's own
           timing, unchanged. */
        this.bodies.forEach((b, i) => {
          const ph = i * 2.1;
          const bs = i ? Math.sin(t * 0.62 + ph) : breath;
          const swell = 1 + Math.sin(t * 0.62 + 0.5 + ph) * 0.016 * amp;
          const [csx, csy, csz] = b.capScale;
          b.capGroup.scale.set(csx * swell, csy * (1 + bs * 0.026 * amp), csz * swell);
          b.stem.scale.y = 1 - bs * 0.012 * amp;
        });
        /* The 0.34 here is the reference's own drift off the 0.30 it booted at
           — the camera settles a little right of where it starts. Kept as an
           offset so every specimen drifts by the same amount from its own
           framing instead of all of them snapping to the hero's. */
        this.camera.position.x = V.camera.x + 0.04 + p.x * 0.16 * amp;
        this.camera.position.y = V.camera.y - p.y * 0.1 * amp;
        this.camera.lookAt(this.target);
        const pulse = 0.5 + Math.sin(t * 0.38) * 0.5;
        this.halos[0].material.opacity = (0.36 + pulse * 0.2 + g * 0.26) * V.tune.halo;
        this.halos[0].scale.setScalar(4.1 + pulse * 0.22 + g * 0.5);
        this.halos[1].material.opacity = (0.3 + (0.5 + Math.sin(t * 0.55 + 1.1) * 0.5) * 0.22 + g * 0.3) * V.tune.halo;
        this.mats.gillMat.emissiveIntensity = (0.28 + pulse * 0.14 + g * 0.6) * V.tune.glow;
        this.bodies.forEach((b, i) => {
          const p2 = i ? 0.5 + Math.sin(t * 0.38 + i * 2.1) * 0.5 : pulse;
          b.inner.intensity = (1.15 + p2 * 0.45 + g * 1.4) * V.tune.glow;
          b.shellMat.uniforms.uStrength.value = (0.72 + p2 * 0.22 + g * 0.35) * V.tune.shell;
        });
        this.sporeMat.uniforms.uTime.value = t;
      } else {
        if (!this._needsRender) return;
        this.group.rotation.set(0.17, 0.18, -0.03);
        this.group.scale.setScalar(this.baseScale);
        this.bodies.forEach((b) => b.capGroup.scale.set(...b.capScale));
        this.camera.lookAt(this.target);
        this._needsRender = false;
      }

      const r = this.renderer;
      /* Paint the actual mushroom before compiling the bloom pipeline. Shader
         compilation for the four post-processing passes was the visible
         two-to-three second gap on a cold load. The direct material shaders
         are enough for the first real frame; bloom joins on the next frame,
         after the host is already visible. */
      if (!this._firstFramePainted) {
        r.setRenderTarget(null);
        r.clear();
        r.render(this.scene, this.camera);
        this._firstFramePainted = true;
        requestAnimationFrame(() => {
          if (!this._dead) this.dispatchEvent(new CustomEvent('mushroom-ready', { bubbles: true }));
        });
        return;
      }
      r.setRenderTarget(this.rtScene);
      r.clear();
      r.render(this.scene, this.camera);
      const wA = this.rtA.width, hA = this.rtA.height;
      this.pBright.uniforms.tDiffuse.value = this.rtScene.texture;
      this.pass(this.pBright, this.rtA);
      const blur = (src, dst, dx, dy) => {
        this.pBlur.uniforms.tDiffuse.value = src.texture;
        this.pBlur.uniforms.uDir.value.set(dx, dy);
        this.pass(this.pBlur, dst);
      };
      blur(this.rtA, this.rtB, 1 / wA, 0);
      blur(this.rtB, this.rtA, 0, 1 / hA);
      blur(this.rtA, this.rtB, 2.4 / wA, 0);
      blur(this.rtB, this.rtA, 0, 2.4 / hA);
      this.pComp.uniforms.tScene.value = this.rtScene.texture;
      this.pComp.uniforms.tBloom.value = this.rtA.texture;
      this.pass(this.pComp, null);
    };
  }

  customElements.define('mushroom-stage', MushroomStage);
}
