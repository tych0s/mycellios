import { useEffect, useRef } from "react";

type Tip = {
  x: number;
  y: number;
  px: number;
  py: number;
  angle: number;
  generation: number;
  primary: boolean;
  life: number;
  width: number;
  speed: number;
  wander: number;
  bias: number;
  turn: number;
  wait: number;
  alive: boolean;
};

type Junction = { x: number; y: number; radius: number; phase: number; brightness: number };
type Spore = { x: number; y: number; vx: number; vy: number; radius: number; alpha: number; phase: number };
type GpuLight = {
  fromX: number;
  fromY: number;
  toX: number;
  toY: number;
  x: number;
  y: number;
  progress: number;
  speed: number;
  curve: number;
  phase: number;
  radius: number;
  color: number;
};

const FILAMENT = [196, 220, 255] as const;
const GPU_COLORS = ["#5c96ff", "#83b5ff", "#b9d2ff", "#718dff", "#593efe"] as const;

function MyceliumHero() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvasCandidate = canvasRef.current;
    if (!canvasCandidate) return;
    const canvas = canvasCandidate;
    const parentCandidate = canvas.parentElement;
    if (!parentCandidate) return;
    const parent = parentCandidate;

    const contextCandidate = canvas.getContext("2d");
    const trail = document.createElement("canvas");
    const trailContextCandidate = trail.getContext("2d");
    if (!contextCandidate || !trailContextCandidate) return;
    const context: CanvasRenderingContext2D = contextCandidate;
    const trailContext: CanvasRenderingContext2D = trailContextCandidate;

    const sporeSprite = document.createElement("canvas");
    sporeSprite.width = 64;
    sporeSprite.height = 64;
    const spriteContext = sporeSprite.getContext("2d");
    if (!spriteContext) return;
    const spriteGradient = spriteContext.createRadialGradient(32, 32, 0, 32, 32, 32);
    spriteGradient.addColorStop(0, "rgba(255,255,255,1)");
    spriteGradient.addColorStop(0.2, "rgba(224,237,255,.62)");
    spriteGradient.addColorStop(0.48, "rgba(92,150,255,.2)");
    spriteGradient.addColorStop(1, "rgba(120,200,255,0)");
    spriteContext.fillStyle = spriteGradient;
    spriteContext.fillRect(0, 0, 64, 64);

    const gpuSprites = GPU_COLORS.map((color) => {
      const sprite = document.createElement("canvas");
      sprite.width = 72;
      sprite.height = 72;
      const spriteContext = sprite.getContext("2d");
      if (!spriteContext) return sprite;
      const glow = spriteContext.createRadialGradient(36, 36, 0, 36, 36, 36);
      glow.addColorStop(0, "rgba(255,255,255,1)");
      glow.addColorStop(.12, color);
      glow.addColorStop(.32, `${color}88`);
      glow.addColorStop(1, `${color}00`);
      spriteContext.fillStyle = glow;
      spriteContext.fillRect(0, 0, 72, 72);
      return sprite;
    });

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let width = 0;
    let height = 0;
    let dpr = 1;
    let originX = 0;
    let originY = 0;
    let maxRadius = 0;
    let tips: Tip[] = [];
    let junctions: Junction[] = [];
    let spores: Spore[] = [];
    let gpuLights: GpuLight[] = [];
    let phase: "growing" | "weaving" = "growing";
    let breath = 0;
    let opacity = 0;
    let lastFrameTime = 0;
    let growthAccumulator = 0;
    let trailFadeAccumulator = 0;
    let animationFrame = 0;
    let visible = true;
    let destroyed = false;

    function makeTip(x: number, y: number, angle: number, generation: number, primary: boolean): Tip {
      return {
        x, y, px: x, py: y, angle, generation, primary,
        life: primary ? Number.POSITIVE_INFINITY : Math.round(48 + Math.random() * 115 * Math.max(.3, 1 - generation * .1)),
        width: primary ? 2.15 : Math.max(.35, 1.75 - generation * .3),
        speed: 1.24 * (primary ? .86 + Math.random() * .58 : .56 + Math.random() * .54),
        wander: primary ? .055 : .14 + Math.random() * .15,
        bias: primary ? .052 : .015,
        turn: (Math.random() - .5) * .1,
        wait: primary ? Math.floor(Math.random() * Math.random() * 18) : Math.floor(Math.random() * 20),
        alive: true,
      };
    }

    function reset() {
      trailContext.clearRect(0, 0, width, height);
      originX = width < 760 ? width * .54 : width * .7;
      originY = width < 760 ? height * .79 : height * .52;
      maxRadius = Math.max(
        Math.hypot(originX, originY),
        Math.hypot(width - originX, originY),
        Math.hypot(originX, height - originY),
        Math.hypot(width - originX, height - originY),
      ) * .98;
      tips = [];
      junctions = [];
      spores = [];
      gpuLights = [];
      phase = "growing";
      breath = 0;
      opacity = 0;
      lastFrameTime = 0;
      growthAccumulator = 0;
      trailFadeAccumulator = 0;

      const primaryCount = width < 760 ? 16 : 22;
      for (let index = 0; index < primaryCount; index += 1) {
        const angle = (index / primaryCount) * Math.PI * 2 + (Math.random() - .5) * .86;
        tips.push(makeTip(originX, originY, angle, 0, true));
      }

      const sporeCount = Math.round((width * height) / 25_000);
      for (let index = 0; index < sporeCount; index += 1) {
        spores.push({
          x: Math.random() * width,
          y: Math.random() * height,
          vx: (Math.random() - .5) * .12,
          vy: (Math.random() - .5) * .12,
          radius: .5 + Math.random() * 1.4,
          alpha: .05 + Math.random() * .22,
          phase: Math.random() * Math.PI * 2,
        });
      }
    }

    function resize() {
      const nextWidth = Math.round(parent.clientWidth);
      const nextHeight = Math.round(parent.clientHeight);
      if (!nextWidth || !nextHeight || (nextWidth === width && nextHeight === height)) return;
      width = nextWidth;
      height = nextHeight;
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      trail.width = Math.round(width * dpr);
      trail.height = Math.round(height * dpr);
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      trailContext.setTransform(dpr, 0, 0, dpr, 0, 0);
      context.imageSmoothingEnabled = true;
      context.imageSmoothingQuality = "high";
      trailContext.imageSmoothingEnabled = true;
      trailContext.imageSmoothingQuality = "high";
      reset();

      if (reducedMotion) {
        for (let index = 0; index < 460 && phase === "growing"; index += 1) step();
        opacity = 1;
        renderFrame();
      }
    }

    function drawSegment(tip: Tip) {
      const [red, green, blue] = FILAMENT;
      const violetMix = Math.min(.22, tip.generation * .035);
      const distanceFromCore = Math.hypot(tip.x - originX, tip.y - originY);
      const emergence = Math.min(1, distanceFromCore / (Math.min(width, height) * .18));
      const alpha = (tip.primary ? .38 : .3) * (.06 + emergence * .94);
      const filamentRed = Math.round(red - 45 * violetMix);
      const filamentGreen = Math.round(green - 42 * violetMix);
      trailContext.globalCompositeOperation = "lighter";
      trailContext.lineCap = "round";
      trailContext.lineJoin = "round";

      trailContext.strokeStyle = `rgba(${filamentRed},${filamentGreen},${blue},${alpha * .2})`;
      trailContext.lineWidth = Math.max(1.15, tip.width * 1.85);
      trailContext.beginPath();
      trailContext.moveTo(tip.px, tip.py);
      trailContext.lineTo(tip.x, tip.y);
      trailContext.stroke();

      trailContext.strokeStyle = `rgba(${Math.min(255, filamentRed + 24)},${Math.min(255, filamentGreen + 18)},255,${Math.min(.78, alpha * 1.5)})`;
      trailContext.lineWidth = Math.max(.52, tip.width * .62);
      trailContext.beginPath();
      trailContext.moveTo(tip.px, tip.py);
      trailContext.lineTo(tip.x, tip.y);
      trailContext.stroke();
    }

    function step() {
      const newTips: Tip[] = [];
      for (const tip of tips) {
        if (!tip.alive) continue;
        if (tip.wait > 0) {
          tip.wait -= 1;
          continue;
        }
        tip.px = tip.x;
        tip.py = tip.y;
        tip.turn += (Math.random() - .5) * .09;
        tip.turn *= .92;
        tip.turn = Math.max(-.24, Math.min(.24, tip.turn));
        tip.angle += tip.turn + (Math.random() - .5) * tip.wander * .4;
        const outward = Math.atan2(tip.y - originY, tip.x - originX);
        const difference = Math.atan2(Math.sin(outward - tip.angle), Math.cos(outward - tip.angle));
        tip.angle += difference * tip.bias;
        tip.x += Math.cos(tip.angle) * tip.speed;
        tip.y += Math.sin(tip.angle) * tip.speed;
        drawSegment(tip);
        tip.life -= 1;

        const radius = Math.hypot(tip.x - originX, tip.y - originY);
        if (tip.life <= 0 || radius > maxRadius) {
          tip.alive = false;
          continue;
        }

        const branchChance = .046 * (tip.primary ? 1.5 : 1) * (radius < maxRadius * .35 ? 1.38 : 1);
        if (Math.random() < branchChance && tip.generation < 7 && tips.length + newTips.length < 380) {
          const offset = (Math.random() < .5 ? -1 : 1) * (.28 + Math.random() * .56);
          newTips.push(makeTip(tip.x, tip.y, tip.angle + offset, tip.generation + 1, false));
          if (junctions.length < 520) {
            junctions.push({
              x: tip.x,
              y: tip.y,
              radius: .7 + Math.random() * 1.8,
              phase: Math.random() * Math.PI * 2,
              brightness: .4 + Math.random() * .6,
            });
          }
        }
      }
      tips.push(...newTips);
      tips = tips.filter((tip) => tip.alive);
      if (!tips.length) seedSlowWeaving();
    }

    function seedSlowWeaving() {
      phase = "weaving";
      const innerRadius = Math.min(width, height) * .13;
      const candidates = junctions.filter((junction) => {
        const radius = Math.hypot(junction.x - originX, junction.y - originY);
        return radius > innerRadius && radius < maxRadius * .82;
      });
      const sourcePool = candidates.length ? candidates : junctions;
      const seedCount = width < 760 ? 1 : 2;

      for (let index = 0; index < seedCount; index += 1) {
        const source = sourcePool[Math.floor(Math.random() * sourcePool.length)];
        const x = source?.x ?? originX;
        const y = source?.y ?? originY;
        const outward = Math.atan2(y - originY, x - originX);
        const tip = makeTip(x, y, outward + (Math.random() - .5) * 2.15, 2 + Math.floor(Math.random() * 3), false);
        tip.life = Math.round(80 + Math.random() * 125);
        tip.speed *= .58;
        tip.wander *= .7;
        tip.bias = .006;
        tip.wait = Math.round(8 + Math.random() * 24);
        tips.push(tip);
      }
    }

    function nearbyJunction(x: number, y: number): Junction {
      let selected = junctions[Math.floor(Math.random() * junctions.length)]!;
      let bestScore = Number.POSITIVE_INFINITY;
      for (let attempt = 0; attempt < 36; attempt += 1) {
        const candidate = junctions[Math.floor(Math.random() * junctions.length)]!;
        const distance = Math.hypot(candidate.x - x, candidate.y - y);
        const score = Math.abs(distance - 150) + Math.random() * 55;
        if (distance > 45 && distance < 310 && score < bestScore) {
          selected = candidate;
          bestScore = score;
        }
      }
      return selected;
    }

    function retargetGpu(light: GpuLight) {
      light.fromX = light.toX;
      light.fromY = light.toY;
      const target = nearbyJunction(light.fromX, light.fromY);
      light.toX = target.x;
      light.toY = target.y;
      light.progress = 0;
      light.speed = .0013 + Math.random() * .0021;
      light.curve = (Math.random() - .5) * 46;
    }

    function ensureGpuLights() {
      if (junctions.length < 48) return;
      const targetCount = width < 760 ? 9 : 16;
      while (gpuLights.length < targetCount) {
        const start = junctions[Math.floor(Math.random() * junctions.length)]!;
        const target = nearbyJunction(start.x, start.y);
        gpuLights.push({
          fromX: start.x,
          fromY: start.y,
          toX: target.x,
          toY: target.y,
          x: start.x,
          y: start.y,
          progress: Math.random(),
          speed: .0013 + Math.random() * .0021,
          curve: (Math.random() - .5) * 46,
          phase: Math.random() * Math.PI * 2,
          radius: 2.25 + Math.random() * 1.65,
          color: gpuLights.length % GPU_COLORS.length,
        });
      }
    }

    function drawGpuLights() {
      ensureGpuLights();
      context.save();
      context.globalCompositeOperation = "lighter";
      for (const light of gpuLights) {
        light.progress += light.speed;
        light.phase += .035;
        if (light.progress >= 1) retargetGpu(light);
        const eased = .5 - Math.cos(light.progress * Math.PI) / 2;
        const dx = light.toX - light.fromX;
        const dy = light.toY - light.fromY;
        const distance = Math.max(1, Math.hypot(dx, dy));
        const curve = Math.sin(eased * Math.PI) * light.curve;
        light.x = light.fromX + dx * eased + (-dy / distance) * curve;
        light.y = light.fromY + dy * eased + (dx / distance) * curve;
        const shimmer = .78 + .22 * Math.sin(light.phase);
        const halo = 17 + light.radius * 8;
        context.globalAlpha = opacity * shimmer;
        context.drawImage(gpuSprites[light.color]!, light.x - halo, light.y - halo, halo * 2, halo * 2);
        context.fillStyle = GPU_COLORS[light.color]!;
        context.beginPath();
        context.arc(light.x, light.y, light.radius, 0, Math.PI * 2);
        context.fill();
        context.globalAlpha = opacity * shimmer * .7;
        context.strokeStyle = GPU_COLORS[light.color]!;
        context.lineWidth = .65;
        context.beginPath();
        context.arc(light.x, light.y, light.radius * 2.35, 0, Math.PI * 2);
        context.stroke();
      }
      context.restore();
    }

    function renderFrame(frameDelta = 0) {
      const [red, green, blue] = FILAMENT;
      context.clearRect(0, 0, width, height);
      breath += frameDelta * .00007;
      const pulse = .89 + .11 * Math.sin(breath);
      const driftX = Math.cos(breath * .37) * 2.2;
      const driftY = Math.sin(breath * .29) * 1.8;
      const crispDriftX = Math.round(driftX * dpr) / dpr;
      const crispDriftY = Math.round(driftY * dpr) / dpr;

      context.save();
      context.globalAlpha = .94 * pulse * opacity;
      context.translate(crispDriftX, crispDriftY);
      context.drawImage(trail, 0, 0, trail.width, trail.height, 0, 0, width, height);
      context.restore();

      const coreRadius = Math.min(width, height) * (width < 760 ? .085 : .095);
      context.save();
      context.globalCompositeOperation = "lighter";
      for (let index = 0; index < junctions.length; index += 6) {
        const junction = junctions[index]!;
        const glow = .55 + .45 * Math.sin(breath * 1.25 + junction.phase);
        const radius = junction.radius * (2.4 + glow * 1.3);
        context.globalAlpha = Math.min(1, junction.brightness * glow * opacity);
        context.drawImage(sporeSprite, junction.x - radius, junction.y - radius, radius * 2, radius * 2);
      }
      if (tips.length) {
        for (const tip of tips) {
          if (!tip.alive || tip.wait > 0) continue;
          const radius = 2.5 + tip.width * 1.25;
          context.globalAlpha = .88 * opacity;
          context.drawImage(sporeSprite, tip.x - radius, tip.y - radius, radius * 2, radius * 2);
        }
      }
      context.restore();

      const coreGradient = context.createRadialGradient(originX, originY, 0, originX, originY, coreRadius * 2.35);
      coreGradient.addColorStop(0, `rgba(119,167,255,${.48 * opacity})`);
      coreGradient.addColorStop(.18, `rgba(76,132,242,${.27 * opacity})`);
      coreGradient.addColorStop(.42, `rgba(63,124,255,${.14 * opacity})`);
      coreGradient.addColorStop(.7, `rgba(88,112,196,${.06 * opacity})`);
      coreGradient.addColorStop(1, "rgba(17,23,47,0)");
      context.save();
      context.fillStyle = coreGradient;
      context.beginPath();
      context.arc(originX, originY, coreRadius * 2.35, 0, Math.PI * 2);
      context.fill();
      const nucleusRadius = coreRadius * .58;
      const traceHexagon = (radius: number, rotation = Math.PI / 6) => {
        context.beginPath();
        for (let point = 0; point < 6; point += 1) {
          const angle = rotation + point / 6 * Math.PI * 2;
          const x = originX + Math.cos(angle) * radius;
          const y = originY + Math.sin(angle) * radius;
          if (point === 0) context.moveTo(x, y);
          else context.lineTo(x, y);
        }
        context.closePath();
      };
      const nucleusGradient = context.createLinearGradient(
        originX - nucleusRadius,
        originY - nucleusRadius,
        originX + nucleusRadius,
        originY + nucleusRadius,
      );
      nucleusGradient.addColorStop(0, "rgba(198,220,255,.94)");
      nucleusGradient.addColorStop(.28, "rgba(91,151,255,.9)");
      nucleusGradient.addColorStop(.7, "rgba(63,83,213,.86)");
      nucleusGradient.addColorStop(1, "rgba(46,31,127,.9)");
      context.fillStyle = nucleusGradient;
      traceHexagon(nucleusRadius);
      context.fill();
      context.strokeStyle = `rgba(211,228,255,${.72 * opacity})`;
      context.lineWidth = 1.1;
      context.stroke();

      context.globalCompositeOperation = "lighter";
      for (let shard = 0; shard < 3; shard += 1) {
        const shardY = originY + (shard - 1) * nucleusRadius * .42;
        const shardWidth = nucleusRadius * (1.08 - shard * .08);
        const shardHeight = Math.max(4, nucleusRadius * .16);
        const inset = shardHeight * .62;
        const shardGradient = context.createLinearGradient(originX - shardWidth, shardY, originX + shardWidth, shardY);
        shardGradient.addColorStop(0, `rgba(132,181,255,${.48 * opacity})`);
        shardGradient.addColorStop(.52, `rgba(226,239,255,${.72 * opacity})`);
        shardGradient.addColorStop(1, `rgba(113,82,255,${.6 * opacity})`);
        context.fillStyle = shardGradient;
        context.beginPath();
        context.moveTo(originX - shardWidth / 2 + inset, shardY - shardHeight);
        context.lineTo(originX + shardWidth / 2, shardY - shardHeight);
        context.lineTo(originX + shardWidth / 2 - inset, shardY + shardHeight);
        context.lineTo(originX - shardWidth / 2, shardY + shardHeight);
        context.closePath();
        context.fill();
        context.strokeStyle = `rgba(224,238,255,${.55 * opacity})`;
        context.lineWidth = .7;
        context.stroke();
      }

      context.strokeStyle = `rgba(221,236,255,${.76 * pulse * opacity})`;
      context.lineWidth = 1.1;
      context.beginPath();
      context.moveTo(originX, originY - nucleusRadius * .72);
      context.lineTo(originX, originY + nucleusRadius * .72);
      context.stroke();
      context.globalCompositeOperation = "lighter";
      for (let ring = 0; ring < 3; ring += 1) {
        context.strokeStyle = ring === 2 ? `rgba(131,181,255,${.22 * pulse * opacity})` : `rgba(${red},${green},${blue},${(.42 - ring * .1) * pulse * opacity})`;
        context.lineWidth = ring === 0 ? 1.8 : .8;
        context.setLineDash(ring === 0 ? [] : [3 + ring * 2, 7 + ring * 3]);
        context.lineDashOffset = (ring % 2 ? 1 : -1) * breath * (5 + ring * 2);
        traceHexagon(coreRadius * (.55 + ring * .48), Math.PI / 6 + ring * Math.PI / 12);
        context.stroke();
      }
      context.setLineDash([]);
      for (let port = 0; port < 6; port += 1) {
        const angle = Math.PI / 6 + port / 6 * Math.PI * 2;
        const inner = coreRadius * .58;
        const outer = coreRadius * .78;
        context.strokeStyle = `rgba(170,205,255,${.5 * pulse * opacity})`;
        context.lineWidth = 1;
        context.beginPath();
        context.moveTo(originX + Math.cos(angle) * inner, originY + Math.sin(angle) * inner);
        context.lineTo(originX + Math.cos(angle) * outer, originY + Math.sin(angle) * outer);
        context.stroke();
      }
      for (let satellite = 0; satellite < 8; satellite += 1) {
        const angle = satellite / 8 * Math.PI * 2 + breath * .08 * (satellite % 2 ? 1 : -1);
        const orbit = coreRadius * (satellite % 2 ? .78 : 1.12);
        const x = originX + Math.cos(angle) * orbit;
        const y = originY + Math.sin(angle) * orbit;
        const size = satellite % 3 === 0 ? 7 : 5;
        context.globalAlpha = .55 * opacity;
        context.drawImage(gpuSprites[satellite % gpuSprites.length]!, x - size, y - size, size * 2, size * 2);
      }
      context.restore();

      drawGpuLights();

      context.save();
      context.globalCompositeOperation = "lighter";
      for (const spore of spores) {
        spore.x += spore.vx;
        spore.y += spore.vy;
        spore.phase += .02;
        if (spore.x < 0) spore.x += width;
        if (spore.x > width) spore.x -= width;
        if (spore.y < 0) spore.y += height;
        if (spore.y > height) spore.y -= height;
        const radius = spore.radius * 3;
        context.globalAlpha = Math.min(1, (.6 + .4 * Math.sin(spore.phase)) * spore.alpha * opacity);
        context.drawImage(sporeSprite, spore.x - radius, spore.y - radius, radius * 2, radius * 2);
      }
      context.restore();
    }

    function schedule() {
      if (!destroyed && visible && !reducedMotion && !animationFrame) animationFrame = window.requestAnimationFrame(loop);
    }

    function loop(frameTime: number) {
      animationFrame = 0;
      if (!visible || destroyed) return;

      const frameDelta = lastFrameTime ? Math.min(64, frameTime - lastFrameTime) : 16.67;
      lastFrameTime = frameTime;
      growthAccumulator += frameDelta;
      trailFadeAccumulator += frameDelta;
      const growthStepDuration = phase === "growing" ? 46 : 92;

      while (growthAccumulator >= growthStepDuration) {
        step();
        growthAccumulator -= growthStepDuration;
      }

      if (trailFadeAccumulator >= 1_000) {
        const fadePasses = Math.floor(trailFadeAccumulator / 1_000);
        trailFadeAccumulator -= fadePasses * 1_000;
        trailContext.save();
        trailContext.globalCompositeOperation = "destination-out";
        trailContext.fillStyle = `rgba(0,0,0,${Math.min(.0012, fadePasses * .00016)})`;
        trailContext.fillRect(0, 0, width, height);
        trailContext.restore();
      }

      opacity = Math.min(1, opacity + frameDelta / 3_600);
      renderFrame(frameDelta);
      schedule();
    }

    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(parent);
    const visibilityObserver = new IntersectionObserver(([entry]) => {
      visible = entry?.isIntersecting ?? true;
      if (visible) schedule();
    });
    visibilityObserver.observe(parent);
    resize();
    schedule();

    return () => {
      destroyed = true;
      resizeObserver.disconnect();
      visibilityObserver.disconnect();
      window.cancelAnimationFrame(animationFrame);
    };
  }, []);

  return (
    <div className="mycelium-hero-canvas" role="img" aria-label="Living mycelium growing outward and connecting into a distributed network">
      <canvas ref={canvasRef} aria-hidden="true" />
      <div className="mycelium-core-overlay" aria-hidden="true"><span /><span /><span /><i /></div>
      <div className="mycelium-growth-status" aria-hidden="true"><i /><span>DISTRIBUTED MODEL · LIVE</span><b>SHARDS ROUTED ACROSS NODES</b></div>
      <div className="gpu-coordinate gpu-coordinate-a" aria-hidden="true"><i /><span>RTX 4090</span><b>24 GB</b></div>
      <div className="gpu-coordinate gpu-coordinate-b" aria-hidden="true"><i /><span>APPLE M3</span><b>36 GB</b></div>
      <div className="gpu-coordinate gpu-coordinate-c" aria-hidden="true"><i /><span>RTX 3060</span><b>12 GB</b></div>
      <div className="network-whisper" aria-hidden="true"><span>ROUTE 7FA3</span><i /><b>MODEL SHARD MOVING</b></div>
    </div>
  );
}

export { MyceliumHero };
