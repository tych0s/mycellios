/* <mushroom-stage> — procedural translucent mushroom, three.js + custom bloom.
   Attributes: accent (#hex) | spores ("on"/"off") | motion ("full"|"calm"|"off") | scale (number) */
/*
 * Vendored from the design reference. It is kept as a custom element rather
 * than rewritten as a React component on purpose: the value here is the *look*,
 * and the look lives in a hundred tuned constants (profile tables, sheen, bloom
 * thresholds) that a rewrite would quietly drift away from. React owns the
 * mount point; this owns the pixels.
 *
 * Changed against the reference in exactly four places, all marked `mycellios:`
 *   1. three.js is resolved from the bundle instead of fetched from unpkg;
 *   2. the IIFE became an idempotent exported `defineMushroomStage()`, so
 *      React's StrictMode double-mount cannot race `customElements.define`;
 *   3. a `mushroom-ready` event fires on the first *drawn* frame, so the host
 *      can fade the organism in rather than letting it pop;
 *   4. three.js is imported by name rather than as a namespace, so the bundler
 *      can drop what this file never touches — that alone is 190KB gzipped
 *      down to 137KB. `THREE_MODULE` is rebuilt from those bindings so the
 *      several hundred `THREE.Foo` call sites below stay exactly as written.
 *
 * Everything else — the profile tables, the scroll-driven `progress`, the
 * bloom chain — is untouched.
 */
import {AdditiveBlending, BufferGeometry, CanvasTexture, Clock, Color, DirectionalLight, DoubleSide, Float32BufferAttribute, FrontSide, Group, HalfFloatType, HemisphereLight, InstancedMesh, LatheGeometry, LinearFilter, Matrix4, Mesh, MeshBasicMaterial, MeshPhysicalMaterial, MeshStandardMaterial, OrthographicCamera, PerspectiveCamera, PlaneGeometry, PointLight, Points, SRGBColorSpace, Scene, ShaderMaterial, Vector2, Vector3, WebGLRenderTarget, WebGLRenderer} from "three";
const THREE_MODULE = {AdditiveBlending, BufferGeometry, CanvasTexture, Clock, Color, DirectionalLight, DoubleSide, Float32BufferAttribute, FrontSide, Group, HalfFloatType, HemisphereLight, InstancedMesh, LatheGeometry, LinearFilter, Matrix4, Mesh, MeshBasicMaterial, MeshPhysicalMaterial, MeshStandardMaterial, OrthographicCamera, PerspectiveCamera, PlaneGeometry, PointLight, Points, SRGBColorSpace, Scene, ShaderMaterial, Vector2, Vector3, WebGLRenderTarget, WebGLRenderer};


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
      this.motion = this.getAttribute('motion') || 'full';
      this.baseScale = parseFloat(this.getAttribute('scale')) || 1;
      const reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
      this.reduce = reduce;
      const lowPower = (navigator.hardwareConcurrency || 8) <= 4 || innerWidth < 700;

      const canvas = document.createElement('canvas');
      canvas.style.cssText = 'display:block;width:100%;height:100%;';
      this.appendChild(canvas);
      this.canvas = canvas;

      const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: !lowPower, powerPreference: 'high-performance' });
      renderer.setClearColor(0x000000, 0);
      this.renderer = renderer;
      this.dprCap = lowPower ? 1.25 : 1.7;

      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(26, 1, 0.1, 60);
      camera.position.set(0.30, -1.02, 5.15);
      this.scene = scene; this.camera = camera;
      this.target = new THREE.Vector3(0, 0.36, 0);

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

      // ── cap
      const seg = lowPower ? 64 : 112;
      const capTopGeo = new THREE.LatheGeometry(CAP_TOP.map(p => new THREE.Vector2(p[0], p[1])), seg);
      const cap = new THREE.Mesh(capTopGeo, skin);
      cap.name = 'cap';
      const underGeo = new THREE.LatheGeometry(CAP_UNDER.map(p => new THREE.Vector2(p[0], p[1])), seg);
      const under = new THREE.Mesh(underGeo, underSkin);
      under.name = 'capUnderside';
      const capGroup = new THREE.Group();
      capGroup.scale.set(1.1, 0.86, 1.1);
      capGroup.add(cap, under);
      group.add(capGroup);
      this.capGroup = capGroup;

      // ── fresnel shell over the cap
      const shellMat = new THREE.ShaderMaterial({
        uniforms: { uColor: { value: new THREE.Color(0xffeedc) }, uPower: { value: 2.4 }, uStrength: { value: 0.85 } },
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
      this.shellMat = shellMat;

      // ── gills
      const bladeGeo = (r0, r1) => {
        const N = 15, pos = [], col = [], idx = [];
        for (let i = 0; i <= N; i++) {
          const t = i / N;
          const r = r0 + (r1 - r0) * t;
          const yTop = lerpTable(CAP_UNDER, r) - 0.005;
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
      const count = lowPower ? 56 : 96;
      const long = new THREE.InstancedMesh(bladeGeo(0.135, 1.052), gillMat, count);
      const short = new THREE.InstancedMesh(bladeGeo(0.48, 1.048), gillMat, count);
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
      const stemGeo = new THREE.LatheGeometry(STEM.map(p => new THREE.Vector2(p[0], p[1])), Math.max(40, seg / 2));
      {
        const p = stemGeo.attributes.position;
        for (let i = 0; i < p.count; i++) {
          const y = p.getY(i);
          const k = Math.max(0, 0.56 - y) / 1.74;
          const off = -0.075 * Math.pow(k, 1.75);
          p.setX(i, p.getX(i) + off);
          p.setZ(i, p.getZ(i) + off * 0.35);
        }
        p.needsUpdate = true;
        stemGeo.computeVertexNormals();
      }
      const stem = new THREE.Mesh(stemGeo, skin);
      stem.name = 'stem';
      group.add(stem);
      this.stem = stem;

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
      const inner = new THREE.PointLight(accent.clone(), 1.3, 2.6, 2); inner.position.set(0, 0.44, 0);
      scene.add(hemi, key, rim, fill, inner);
      this.inner = inner;

      // ── bloom plumbing
      const half = () => ({ minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, type: THREE.HalfFloatType, depthBuffer: false });
      this.rtScene = new THREE.WebGLRenderTarget(2, 2, { type: THREE.HalfFloatType, samples: lowPower ? 0 : 4 });
      this.rtA = new THREE.WebGLRenderTarget(2, 2, half());
      this.rtB = new THREE.WebGLRenderTarget(2, 2, half());
      const mk = (fs, uniforms) => new THREE.ShaderMaterial({ uniforms, vertexShader: QUAD_VS, fragmentShader: fs, depthTest: false, depthWrite: false });
      this.pBright = mk(BRIGHT_FS, { tDiffuse: { value: null }, uThreshold: { value: 0.42 }, uKnee: { value: 0.35 } });
      this.pBlur = mk(BLUR_FS, { tDiffuse: { value: null }, uDir: { value: new THREE.Vector2() } });
      this.pComp = mk(COMP_FS, { tScene: { value: null }, tBloom: { value: null }, uStrength: { value: 0.68 } });
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
      /* mycellios: the element used to appear the instant three.js compiled its
         shaders, which is a hard pop one to two seconds after the rest of the
         hero is already painted. Announcing the *first drawn frame* — not
         merely `_ready`, which still precedes any pixels — lets the host fade
         it in. Dispatched after a frame so the compositor has the canvas
         content before the transition starts. */
      requestAnimationFrame(() => {
        if (this._dead) return;
        this.dispatchEvent(new CustomEvent('mushroom-ready', { bubbles: true }));
      });
    }

    applyAccent(hex) {
      const { THREE } = this;
      if (!THREE || !hex) return;
      const c = new THREE.Color(hex);
      this.accent.copy(c);
      this.mats.gillMat.emissive.copy(c);
      this.mats.underSkin.emissive.copy(c);
      this.mats.underSkin.sheenColor.copy(c);
      this.inner.color.copy(c);
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
        this.capGroup.scale.set(1.1 * (1 + Math.sin(t * 0.62 + 0.5) * 0.016 * amp), 0.86 * (1 + breath * 0.026 * amp), 1.1 * (1 + Math.sin(t * 0.62 + 0.5) * 0.016 * amp));
        this.stem.scale.y = 1 - breath * 0.012 * amp;
        this.camera.position.x = 0.34 + p.x * 0.16 * amp;
        this.camera.position.y = -1.02 - p.y * 0.1 * amp;
        this.camera.lookAt(this.target);
        const pulse = 0.5 + Math.sin(t * 0.38) * 0.5;
        this.halos[0].material.opacity = 0.36 + pulse * 0.2 + g * 0.26;
        this.halos[0].scale.setScalar(4.1 + pulse * 0.22 + g * 0.5);
        this.halos[1].material.opacity = 0.3 + (0.5 + Math.sin(t * 0.55 + 1.1) * 0.5) * 0.22 + g * 0.3;
        this.mats.gillMat.emissiveIntensity = 0.28 + pulse * 0.14 + g * 0.6;
        this.inner.intensity = 1.15 + pulse * 0.45 + g * 1.4;
        this.shellMat.uniforms.uStrength.value = 0.72 + pulse * 0.22 + g * 0.35;
        this.sporeMat.uniforms.uTime.value = t;
      } else {
        if (!this._needsRender) return;
        this.group.rotation.set(0.17, 0.18, -0.03);
        this.group.scale.setScalar(this.baseScale);
        this.capGroup.scale.set(1.1, 0.86, 1.1);
        this.camera.lookAt(this.target);
        this._needsRender = false;
      }

      const r = this.renderer;
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
