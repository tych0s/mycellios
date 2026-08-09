import {
  Boxes,
  Cpu,
  Laptop,
  MessageSquareText,
  Monitor,
  Network,
  Server,
  Smartphone,
  Sparkles,
} from "lucide-react";
import type { CSSProperties, ReactNode } from "react";
import { useState } from "react";
import { Player } from "@remotion/player";
import {
  AbsoluteFill,
  Easing,
  Img,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import brandMark from "./assets/mycellios-mark-ivory.png";

const FPS = 30;
const SCENE_FRAMES = 150;
const DURATION = SCENE_FRAMES * 4;
const WIDTH = 1600;
const HEIGHT = 900;

/*
 * mycelium brand tokens, hardcoded from `landing/src/brand-tokens.css`
 * (Remotion renders cannot rely on CSS custom properties). This piece uses
 * the dark variant of the system — `.mycelium-dark` — so the video sits
 * seamlessly inside the forest `.rb-explainer-player` frame: forest base,
 * ivory text, bronze accent, sage secondary.
 */
const colors = {
  paper: "#f8f5f0",
  ivory: "#eee9e2",
  ink: "#1e2a22",
  forest: "#142019",
  forestDeep: "#0e1712",
  bronze: "#ad7a48",
  bronzeLight: "#d9b98c",
  sage: "#70806d",
  sageLight: "#93a18e",
  muted: "#b9c2b6",
  line: "rgba(238, 233, 226, 0.16)",
  panel: "rgba(238, 233, 226, 0.055)",
};

const fonts = {
  display: '"Fraunces", Georgia, serif',
  ui: 'Manrope, system-ui, -apple-system, "Segoe UI", sans-serif',
  mono: '"IBM Plex Mono", ui-monospace, SFMono-Regular, monospace',
};

const clamp = {
  extrapolateLeft: "clamp",
  extrapolateRight: "clamp",
} as const;

function sceneOpacity(frame: number, start: number): number {
  return interpolate(
    frame,
    [start, start + 14, start + SCENE_FRAMES - 18, start + SCENE_FRAMES],
    [0, 1, 1, 0],
    clamp,
  );
}

function enter(frame: number, start: number, delay = 0): number {
  return spring({
    frame: Math.max(0, frame - start - delay),
    fps: FPS,
    config: { damping: 22, stiffness: 95, mass: 0.9 },
  });
}

/*
 * Scenes crossfade (no hard cuts): each scene rises in with a spring and
 * drifts gently upward as it hands over to the next one.
 */
function sceneTransform(frame: number, start: number): CSSProperties {
  const value = enter(frame, start);
  const exit = interpolate(
    frame,
    [start + SCENE_FRAMES - 18, start + SCENE_FRAMES],
    [0, 1],
    clamp,
  );
  return {
    opacity: sceneOpacity(frame, start),
    transform: `translateY(${interpolate(value, [0, 1], [30, 0]) - exit * 16}px)`,
  };
}

/* Slow organic drift, scaled by `amount`, for background and idle motion. */
function breathe(frame: number, phase = 0): number {
  return Math.sin(frame / 72 + phase);
}

function Label({ children }: { children: ReactNode }) {
  return (
    <div
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 12,
        color: colors.bronzeLight,
        fontFamily: fonts.mono,
        fontSize: 17,
        fontWeight: 500,
        letterSpacing: "0.14em",
        textTransform: "uppercase",
      }}
    >
      <span
        style={{
          width: 8,
          height: 8,
          borderRadius: "50%",
          background: colors.bronze,
        }}
      />
      {children}
    </div>
  );
}

function SceneHeading({
  label,
  title,
  copy,
}: {
  label: string;
  title: string;
  copy: string;
}) {
  return (
    <div style={{ maxWidth: 1120, textAlign: "center" }}>
      <Label>{label}</Label>
      <h2
        style={{
          margin: "24px 0 16px",
          color: colors.ivory,
          fontFamily: fonts.display,
          fontSize: 68,
          fontWeight: 400,
          lineHeight: 1.04,
          letterSpacing: "-0.045em",
        }}
      >
        {title}
      </h2>
      <p
        style={{
          maxWidth: 900,
          margin: "0 auto",
          color: colors.muted,
          fontSize: 26,
          lineHeight: 1.5,
        }}
      >
        {copy}
      </p>
    </div>
  );
}

function DeviceCard({
  icon,
  title,
  detail,
  accent,
  style,
}: {
  icon: ReactNode;
  title: string;
  detail: string;
  accent: string;
  style?: CSSProperties;
}) {
  return (
    <div
      style={{
        width: 285,
        minHeight: 132,
        display: "grid",
        gridTemplateColumns: "72px 1fr",
        alignItems: "center",
        gap: 20,
        padding: "24px 26px",
        border: `1px solid ${colors.line}`,
        borderRadius: 24,
        background: colors.panel,
        boxShadow: "0 24px 70px rgba(8, 14, 10, 0.45)",
        ...style,
      }}
    >
      <div
        style={{
          width: 68,
          height: 68,
          display: "grid",
          placeItems: "center",
          border: `1px solid ${accent}66`,
          borderRadius: 20,
          color: accent,
          background: `${accent}14`,
        }}
      >
        {icon}
      </div>
      <div>
        <strong
          style={{
            display: "block",
            color: colors.ivory,
            fontSize: 24,
            fontWeight: 600,
          }}
        >
          {title}
        </strong>
        <span
          style={{
            display: "block",
            marginTop: 8,
            color: colors.sageLight,
            fontSize: 16,
          }}
        >
          {detail}
        </span>
      </div>
    </div>
  );
}

function HardwareScene({ frame }: { frame: number }) {
  const start = 0;
  const cardOne = enter(frame, start, 22);
  const cardTwo = enter(frame, start, 34);
  const cardThree = enter(frame, start, 46);
  const cards = [cardOne, cardTwo, cardThree];

  return (
    <AbsoluteFill
      style={{
        ...sceneTransform(frame, start),
        alignItems: "center",
        justifyContent: "center",
        padding: "120px 120px 90px",
      }}
    >
      <SceneHeading
        label="Step 1 · Your hardware"
        title="The capacity already exists."
        copy="Desktops, laptops and workstations spend much of the day underused."
      />
      <div style={{ display: "flex", gap: 30, marginTop: 58 }}>
        {[
          { icon: <Monitor size={36} />, title: "Desktop", accent: colors.bronzeLight },
          { icon: <Laptop size={36} />, title: "Laptop", accent: colors.sageLight },
          { icon: <Server size={36} />, title: "Workstation", accent: colors.bronze },
        ].map((card, index) => {
          const progress = cards[index] ?? 0;
          const float = breathe(frame, index * 1.7) * 4 * progress;
          return (
            <DeviceCard
              key={card.title}
              icon={card.icon}
              title={card.title}
              detail="Capacity available"
              accent={card.accent}
              style={{
                opacity: progress,
                transform: `translateY(${interpolate(progress, [0, 1], [36, 0]) + float}px)`,
              }}
            />
          );
        })}
      </div>
    </AbsoluteFill>
  );
}

const networkNodes = [
  { x: 300, y: 290, label: "DESKTOP", icon: <Monitor size={30} /> },
  { x: 1300, y: 275, label: "LAPTOP", icon: <Laptop size={30} /> },
  { x: 290, y: 655, label: "WORKSTATION", icon: <Server size={30} /> },
  { x: 1310, y: 650, label: "DEVICE", icon: <Smartphone size={30} /> },
] as const;

function NetworkLine({
  x,
  y,
  progress,
  index,
}: {
  x: number;
  y: number;
  progress: number;
  index: number;
}) {
  const centerX = WIDTH / 2;
  const centerY = HEIGHT / 2 + 35;
  const deltaX = x - centerX;
  const deltaY = y - centerY;
  const length = Math.hypot(deltaX, deltaY);
  const angle = Math.atan2(deltaY, deltaX);
  const packet = ((progress * 1.35 + index * 0.24) % 1 + 1) % 1;

  return (
    <div
      style={{
        position: "absolute",
        left: centerX,
        top: centerY,
        width: length,
        height: 2,
        overflow: "visible",
        transformOrigin: "0 50%",
        transform: `rotate(${angle}rad) scaleX(${progress})`,
        background: `linear-gradient(90deg, ${colors.bronze}b0, ${colors.sageLight}59)`,
        boxShadow: `0 0 16px ${colors.bronze}40`,
      }}
    >
      <span
        style={{
          position: "absolute",
          left: `${packet * 100}%`,
          top: -4,
          width: 10,
          height: 10,
          borderRadius: "50%",
          background: colors.bronzeLight,
          boxShadow: `0 0 14px ${colors.bronzeLight}`,
        }}
      />
    </div>
  );
}

function ConnectScene({ frame }: { frame: number }) {
  const start = SCENE_FRAMES;
  const localFrame = Math.max(0, frame - start);
  const lineProgress = interpolate(localFrame, [24, 72], [0, 1], {
    ...clamp,
    easing: Easing.bezier(0.22, 1, 0.36, 1),
  });
  const hub = enter(frame, start, 12);
  const hubPulse = 1 + breathe(localFrame, 0.6) * 0.015;

  return (
    <AbsoluteFill style={{ ...sceneTransform(frame, start) }}>
      <div style={{ position: "absolute", inset: "88px 100px auto", textAlign: "center" }}>
        <SceneHeading
          label="Step 2 · Connect"
          title="mycellios forms one network."
          copy="It measures the available machines and connects only the nodes that help."
        />
      </div>
      {networkNodes.map((node, index) => (
        <NetworkLine
          key={node.label}
          x={node.x}
          y={node.y}
          progress={lineProgress}
          index={index}
        />
      ))}
      <div
        style={{
          position: "absolute",
          left: WIDTH / 2 - 100,
          top: HEIGHT / 2 - 65,
          width: 200,
          height: 200,
          display: "grid",
          placeItems: "center",
          border: `1.5px solid ${colors.bronze}90`,
          borderRadius: "50%",
          background: `radial-gradient(circle, ${colors.bronze}33, ${colors.forestDeep} 70%)`,
          boxShadow: `0 0 0 34px ${colors.bronze}12, 0 0 90px ${colors.bronze}38`,
          opacity: hub,
          transform: `scale(${interpolate(hub, [0, 1], [0.65, 1]) * hubPulse})`,
        }}
      >
        <div
          style={{
            width: 108,
            height: 108,
            display: "grid",
            placeItems: "center",
            borderRadius: "50%",
            background: colors.paper,
            overflow: "hidden",
          }}
        >
          <Img src={brandMark} style={{ width: 96, height: 96 }} />
        </div>
      </div>
      {networkNodes.map((node, index) => {
        const nodeEnter = enter(frame, start, 28 + index * 8);
        return (
          <div
            key={`${node.label}-card`}
            style={{
              position: "absolute",
              left: node.x - 90,
              top: node.y - 48,
              width: 180,
              height: 96,
              display: "grid",
              gridTemplateColumns: "54px 1fr",
              alignItems: "center",
              gap: 12,
              padding: 16,
              border: `1px solid ${colors.line}`,
              borderRadius: 18,
              color: colors.bronzeLight,
              background: colors.panel,
              opacity: nodeEnter,
              transform: `scale(${interpolate(nodeEnter, [0, 1], [0.72, 1])})`,
            }}
          >
            {node.icon}
            <div>
              <strong
                style={{
                  display: "block",
                  color: colors.ivory,
                  fontFamily: fonts.mono,
                  fontSize: 14,
                  fontWeight: 500,
                  letterSpacing: "0.06em",
                }}
              >
                {node.label}
              </strong>
              <span
                style={{
                  display: "block",
                  marginTop: 5,
                  color: colors.sageLight,
                  fontFamily: fonts.mono,
                  fontSize: 12,
                  letterSpacing: "0.08em",
                }}
              >
                CONNECTED
              </span>
            </div>
          </div>
        );
      })}
    </AbsoluteFill>
  );
}

function Shard({
  name,
  color,
  progress,
}: {
  name: string;
  color: string;
  progress: number;
}) {
  return (
    <div
      style={{
        flex: 1,
        minWidth: 0,
        height: 106,
        display: "grid",
        placeItems: "center",
        border: `1px solid ${color}88`,
        borderRadius: 18,
        color,
        background: `${color}16`,
        fontFamily: fonts.mono,
        fontSize: 21,
        fontWeight: 500,
        letterSpacing: "0.06em",
        opacity: progress,
        transform: `translateY(${interpolate(progress, [0, 1], [30, 0])}px)`,
      }}
    >
      {name}
    </div>
  );
}

function SplitScene({ frame }: { frame: number }) {
  const start = SCENE_FRAMES * 2;
  const first = enter(frame, start, 28);
  const second = enter(frame, start, 42);
  const third = enter(frame, start, 56);

  return (
    <AbsoluteFill
      style={{
        ...sceneTransform(frame, start),
        alignItems: "center",
        justifyContent: "center",
        padding: "92px 120px 86px",
      }}
    >
      <SceneHeading
        label="Step 3 · Distribute"
        title="One model. Split across their memory."
        copy="Each machine stores and computes only its assigned part. No single device needs to hold everything."
      />
      <div
        style={{
          width: 1080,
          display: "grid",
          gridTemplateColumns: "1.25fr 90px 1fr",
          alignItems: "center",
          gap: 30,
          marginTop: 48,
        }}
      >
        <div
          style={{
            padding: 28,
            border: `1px solid ${colors.line}`,
            borderRadius: 26,
            background: colors.panel,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 22 }}>
            <Boxes size={30} color={colors.bronzeLight} />
            <strong
              style={{
                color: colors.ivory,
                fontFamily: fonts.mono,
                fontSize: 19,
                fontWeight: 500,
                letterSpacing: "0.08em",
              }}
            >
              LARGE AI MODEL
            </strong>
          </div>
          <div style={{ display: "flex", gap: 14 }}>
            <Shard name="SHARD A" color={colors.bronzeLight} progress={first} />
            <Shard name="SHARD B" color={colors.sageLight} progress={second} />
            <Shard name="SHARD C" color={colors.bronze} progress={third} />
          </div>
        </div>
        <div style={{ position: "relative", height: 2, background: colors.line }}>
          <span
            style={{
              position: "absolute",
              left: `${interpolate(frame, [start + 35, start + 95], [0, 100], {
                ...clamp,
                easing: Easing.bezier(0.22, 1, 0.36, 1),
              })}%`,
              top: -5,
              width: 12,
              height: 12,
              borderRadius: "50%",
              background: colors.bronzeLight,
              boxShadow: `0 0 16px ${colors.bronzeLight}`,
            }}
          />
        </div>
        <div style={{ display: "grid", gap: 12 }}>
          {[
            [colors.bronzeLight, "NODE A", "Stores shard A"],
            [colors.sageLight, "NODE B", "Stores shard B"],
            [colors.bronze, "NODE C", "Stores shard C"],
          ].map(([color, name, detail], index) => {
            const value = [first, second, third][index] ?? 0;
            return (
              <div
                key={name}
                style={{
                  display: "grid",
                  gridTemplateColumns: "52px 1fr",
                  alignItems: "center",
                  gap: 16,
                  padding: "16px 20px",
                  border: `1px solid ${color}55`,
                  borderRadius: 17,
                  background: colors.panel,
                  opacity: value,
                  transform: `translateX(${interpolate(value, [0, 1], [34, 0])}px)`,
                }}
              >
                <Cpu size={31} color={color} />
                <div>
                  <strong
                    style={{
                      display: "block",
                      color: colors.ivory,
                      fontFamily: fonts.mono,
                      fontSize: 16,
                      fontWeight: 500,
                      letterSpacing: "0.06em",
                    }}
                  >
                    {name}
                  </strong>
                  <span style={{ color: colors.muted, fontSize: 15 }}>{detail}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </AbsoluteFill>
  );
}

function AnswerScene({ frame }: { frame: number }) {
  const start = SCENE_FRAMES * 3;
  const localFrame = Math.max(0, frame - start);
  const answer = "The first words appear while the network keeps computing the rest.";
  const visibleCharacters = Math.floor(
    interpolate(localFrame, [54, 118], [0, answer.length], clamp),
  );
  const response = answer.slice(0, visibleCharacters);
  const promptEnter = enter(frame, start, 22);
  const answerEnter = enter(frame, start, 44);
  const caretVisible = Math.floor(localFrame / 16) % 2 === 0;

  return (
    <AbsoluteFill
      style={{
        ...sceneTransform(frame, start),
        alignItems: "center",
        justifyContent: "center",
        padding: "90px 120px 82px",
      }}
    >
      <SceneHeading
        label="Step 4 · Use it"
        title="One prompt in. One answer out."
        copy="mycellios routes the work through the selected nodes and streams the result back."
      />
      <div style={{ width: 1050, display: "grid", gap: 20, marginTop: 44 }}>
        <div
          style={{
            justifySelf: "end",
            width: 720,
            padding: "22px 26px",
            border: `1px solid ${colors.bronze}70`,
            borderRadius: "24px 24px 5px 24px",
            color: colors.ivory,
            background: `${colors.bronze}24`,
            opacity: promptEnter,
            transform: `translateX(${interpolate(promptEnter, [0, 1], [38, 0])}px)`,
          }}
        >
          <span
            style={{
              display: "block",
              marginBottom: 8,
              color: colors.bronzeLight,
              fontFamily: fonts.mono,
              fontSize: 13,
              fontWeight: 500,
              letterSpacing: "0.1em",
            }}
          >
            YOU
          </span>
          <strong style={{ fontSize: 22, fontWeight: 600 }}>
            Explain the result in plain language.
          </strong>
        </div>
        <div
          style={{
            width: 860,
            display: "grid",
            gridTemplateColumns: "60px 1fr",
            gap: 20,
            padding: "24px 28px",
            border: `1px solid ${colors.line}`,
            borderRadius: "24px 24px 24px 5px",
            background: colors.panel,
            opacity: answerEnter,
            transform: `translateX(${interpolate(answerEnter, [0, 1], [-38, 0])}px)`,
          }}
        >
          <div
            style={{
              width: 58,
              height: 58,
              display: "grid",
              placeItems: "center",
              borderRadius: 17,
              background: colors.paper,
              overflow: "hidden",
            }}
          >
            <Img src={brandMark} style={{ width: 52, height: 52 }} />
          </div>
          <div>
            <span
              style={{
                display: "flex",
                alignItems: "center",
                gap: 9,
                color: colors.bronzeLight,
                fontFamily: fonts.mono,
                fontSize: 13,
                fontWeight: 500,
                letterSpacing: "0.1em",
              }}
            >
              <Sparkles size={16} />
              STREAMING
            </span>
            <p
              style={{
                minHeight: 60,
                margin: "12px 0 0",
                color: colors.ivory,
                fontSize: 22,
                lineHeight: 1.48,
              }}
            >
              {response}
              {visibleCharacters < answer.length && (
                <span style={{ color: colors.bronzeLight, opacity: caretVisible ? 1 : 0 }}>|</span>
              )}
            </p>
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
}

function MycelliosExplainerComposition() {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const activeScene = Math.min(3, Math.floor(frame / SCENE_FRAMES));

  return (
    <AbsoluteFill
      style={{
        overflow: "hidden",
        color: colors.ivory,
        background: colors.forest,
        fontFamily: fonts.ui,
      }}
    >
      {/* Warm organic washes that drift slowly, like light through a canopy. */}
      <div
        style={{
          position: "absolute",
          inset: -80,
          background: `radial-gradient(circle at ${18 + breathe(frame) * 2}% 15%, ${colors.bronze}2e, transparent 34%), radial-gradient(circle at ${82 + breathe(frame, 2.1) * 2}% 80%, ${colors.sage}33, transparent 36%), radial-gradient(circle at 50% 45%, ${colors.forestDeep}, transparent 75%)`,
        }}
      />
      <div
        style={{
          position: "absolute",
          inset: 0,
          opacity: 0.5,
          backgroundImage: `linear-gradient(rgba(238, 233, 226, .045) 1px, transparent 1px), linear-gradient(90deg, rgba(238, 233, 226, .045) 1px, transparent 1px)`,
          backgroundSize: "70px 70px",
          maskImage: "radial-gradient(circle at center, black, transparent 78%)",
        }}
      />

      <div
        style={{
          position: "absolute",
          zIndex: 10,
          left: 46,
          top: 38,
          display: "flex",
          alignItems: "center",
          gap: 14,
        }}
      >
        <div
          style={{
            width: 46,
            height: 46,
            display: "grid",
            placeItems: "center",
            borderRadius: 13,
            background: colors.paper,
            overflow: "hidden",
          }}
        >
          <Img src={brandMark} style={{ width: 42, height: 42 }} />
        </div>
        <div>
          <strong
            style={{
              display: "block",
              fontFamily: fonts.display,
              fontSize: 22,
              fontWeight: 500,
              letterSpacing: "-0.01em",
            }}
          >
            mycellios
          </strong>
          <span
            style={{
              color: colors.sageLight,
              fontFamily: fonts.mono,
              fontSize: 12,
              letterSpacing: "0.12em",
            }}
          >
            HOW IT WORKS
          </span>
        </div>
      </div>

      <HardwareScene frame={frame} />
      <ConnectScene frame={frame} />
      <SplitScene frame={frame} />
      <AnswerScene frame={frame} />

      <div
        style={{
          position: "absolute",
          zIndex: 12,
          left: 50,
          right: 50,
          bottom: 30,
          display: "grid",
          gridTemplateColumns: "repeat(4, 1fr)",
          gap: 12,
        }}
      >
        {["CAPACITY", "NETWORK", "MODEL", "ANSWER"].map((label, index) => {
          const sceneStart = index * SCENE_FRAMES;
          const sceneProgress = interpolate(
            frame,
            [sceneStart, sceneStart + SCENE_FRAMES],
            [0, 100],
            clamp,
          );
          return (
            <div key={label}>
              <div
                style={{
                  height: 3,
                  overflow: "hidden",
                  borderRadius: 999,
                  background: "rgba(238, 233, 226, .12)",
                }}
              >
                <span
                  style={{
                    display: "block",
                    width: `${index < activeScene ? 100 : index === activeScene ? sceneProgress : 0}%`,
                    height: "100%",
                    borderRadius: 999,
                    background: index === activeScene ? colors.bronzeLight : colors.sage,
                  }}
                />
              </div>
              <span
                style={{
                  display: "block",
                  marginTop: 8,
                  color: index === activeScene ? colors.ivory : colors.sage,
                  fontFamily: fonts.mono,
                  fontSize: 11,
                  fontWeight: 500,
                  letterSpacing: "0.12em",
                }}
              >
                0{index + 1} · {label}
              </span>
            </div>
          );
        })}
      </div>

      <div
        style={{
          position: "absolute",
          right: 48,
          top: 48,
          color: colors.sage,
          fontFamily: fonts.mono,
          fontSize: 13,
          fontVariantNumeric: "tabular-nums",
          letterSpacing: "0.08em",
        }}
      >
        {Math.min(20, Math.floor(frame / FPS) + 1)} / {Math.floor(durationInFrames / FPS)} SEC
      </div>
    </AbsoluteFill>
  );
}

export default function ExplainerVideo() {
  /*
   * Reduced-motion: the Player API has no built-in media-query handling, so
   * autoplay is gated here. Users with reduced motion get a paused first
   * frame and can still start playback manually via the controls.
   */
  const [allowAutoPlay] = useState(
    () =>
      typeof window === "undefined" ||
      !window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );

  return (
    <Player
      component={MycelliosExplainerComposition}
      durationInFrames={DURATION}
      compositionWidth={WIDTH}
      compositionHeight={HEIGHT}
      fps={FPS}
      autoPlay={allowAutoPlay}
      loop
      controls
      acknowledgeRemotionLicense
      style={{
        width: "100%",
        aspectRatio: "16 / 9",
        background: colors.forest,
      }}
    />
  );
}
