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
import brandIcon from "./assets/mycellios-app-icon-v2.png";

const FPS = 30;
const SCENE_FRAMES = 150;
const DURATION = SCENE_FRAMES * 4;
const WIDTH = 1600;
const HEIGHT = 900;

const colors = {
  ink: "#f5f7ff",
  muted: "#a9b6d2",
  dim: "#6f7d9c",
  line: "rgba(150, 181, 255, 0.2)",
  blue: "#4f8dff",
  cyan: "#67e8c4",
  violet: "#9b8cff",
  panel: "rgba(18, 29, 61, 0.88)",
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
    config: { damping: 18, stiffness: 105, mass: 0.85 },
  });
}

function sceneTransform(frame: number, start: number): CSSProperties {
  const value = enter(frame, start);
  return {
    opacity: sceneOpacity(frame, start),
    transform: `translateY(${interpolate(value, [0, 1], [32, 0])}px)`,
  };
}

function Label({ children }: { children: ReactNode }) {
  return (
    <div
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 12,
        color: colors.cyan,
        fontSize: 19,
        fontWeight: 750,
        letterSpacing: "0.12em",
        textTransform: "uppercase",
      }}
    >
      <span
        style={{
          width: 9,
          height: 9,
          borderRadius: "50%",
          background: colors.cyan,
          boxShadow: `0 0 20px ${colors.cyan}`,
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
          color: colors.ink,
          fontSize: 70,
          fontWeight: 420,
          lineHeight: 1.02,
          letterSpacing: "-0.055em",
        }}
      >
        {title}
      </h2>
      <p
        style={{
          maxWidth: 900,
          margin: "0 auto",
          color: colors.muted,
          fontSize: 27,
          lineHeight: 1.45,
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
        boxShadow: "0 24px 70px rgba(2, 7, 25, 0.3)",
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
          background: `${accent}16`,
        }}
      >
        {icon}
      </div>
      <div>
        <strong
          style={{
            display: "block",
            color: colors.ink,
            fontSize: 24,
            fontWeight: 720,
          }}
        >
          {title}
        </strong>
        <span
          style={{
            display: "block",
            marginTop: 8,
            color: colors.muted,
            fontSize: 17,
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
        <DeviceCard
          icon={<Monitor size={36} />}
          title="Desktop"
          detail="Capacity available"
          accent={colors.blue}
          style={{
            opacity: cardOne,
            transform: `translateY(${interpolate(cardOne, [0, 1], [38, 0])}px)`,
          }}
        />
        <DeviceCard
          icon={<Laptop size={36} />}
          title="Laptop"
          detail="Capacity available"
          accent={colors.cyan}
          style={{
            opacity: cardTwo,
            transform: `translateY(${interpolate(cardTwo, [0, 1], [38, 0])}px)`,
          }}
        />
        <DeviceCard
          icon={<Server size={36} />}
          title="Workstation"
          detail="Capacity available"
          accent={colors.violet}
          style={{
            opacity: cardThree,
            transform: `translateY(${interpolate(cardThree, [0, 1], [38, 0])}px)`,
          }}
        />
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
        background: "linear-gradient(90deg, rgba(79,141,255,.62), rgba(103,232,196,.35))",
        boxShadow: "0 0 18px rgba(79,141,255,.3)",
      }}
    >
      <span
        style={{
          position: "absolute",
          left: `${packet * 100}%`,
          top: -5,
          width: 11,
          height: 11,
          borderRadius: "50%",
          background: colors.cyan,
          boxShadow: `0 0 18px ${colors.cyan}`,
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
    easing: Easing.out(Easing.cubic),
  });
  const hub = enter(frame, start, 12);

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
          border: `2px solid ${colors.blue}88`,
          borderRadius: "50%",
          background: "radial-gradient(circle, rgba(52,94,190,.9), rgba(20,30,67,.96) 68%)",
          boxShadow: "0 0 0 34px rgba(79,141,255,.07), 0 0 90px rgba(79,141,255,.28)",
          opacity: hub,
          transform: `scale(${interpolate(hub, [0, 1], [0.65, 1])})`,
        }}
      >
        <Img src={brandIcon} style={{ width: 82, height: 82 }} />
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
              color: colors.blue,
              background: colors.panel,
              opacity: nodeEnter,
              transform: `scale(${interpolate(nodeEnter, [0, 1], [0.72, 1])})`,
            }}
          >
            {node.icon}
            <div>
              <strong style={{ display: "block", color: colors.ink, fontSize: 16 }}>{node.label}</strong>
              <span style={{ display: "block", marginTop: 5, color: colors.cyan, fontSize: 13 }}>CONNECTED</span>
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
        background: `${color}18`,
        fontSize: 25,
        fontWeight: 800,
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
            <Boxes size={30} color={colors.blue} />
            <strong style={{ color: colors.ink, fontSize: 23 }}>LARGE AI MODEL</strong>
          </div>
          <div style={{ display: "flex", gap: 14 }}>
            <Shard name="SHARD A" color={colors.blue} progress={first} />
            <Shard name="SHARD B" color={colors.cyan} progress={second} />
            <Shard name="SHARD C" color={colors.violet} progress={third} />
          </div>
        </div>
        <div style={{ position: "relative", height: 2, background: colors.line }}>
          <span
            style={{
              position: "absolute",
              left: `${interpolate(frame, [start + 35, start + 95], [0, 100], clamp)}%`,
              top: -6,
              width: 13,
              height: 13,
              borderRadius: "50%",
              background: colors.cyan,
              boxShadow: `0 0 18px ${colors.cyan}`,
            }}
          />
        </div>
        <div style={{ display: "grid", gap: 12 }}>
          {[
            [colors.blue, "NODE A", "Stores shard A"],
            [colors.cyan, "NODE B", "Stores shard B"],
            [colors.violet, "NODE C", "Stores shard C"],
          ].map(([color, name, detail], index) => {
            const value = [first, second, third][index];
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
                  <strong style={{ display: "block", color: colors.ink, fontSize: 18 }}>{name}</strong>
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
            border: `1px solid ${colors.blue}77`,
            borderRadius: "24px 24px 5px 24px",
            color: colors.ink,
            background: "rgba(58, 102, 205, .28)",
            opacity: promptEnter,
            transform: `translateX(${interpolate(promptEnter, [0, 1], [38, 0])}px)`,
          }}
        >
          <span style={{ display: "block", marginBottom: 8, color: colors.blue, fontSize: 14, fontWeight: 800 }}>YOU</span>
          <strong style={{ fontSize: 22, fontWeight: 600 }}>Explain the result in plain language.</strong>
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
              background: "rgba(79, 141, 255, .15)",
            }}
          >
            <Img src={brandIcon} style={{ width: 42, height: 42 }} />
          </div>
          <div>
            <span style={{ display: "flex", alignItems: "center", gap: 9, color: colors.cyan, fontSize: 14, fontWeight: 800 }}>
              <Sparkles size={16} />
              STREAMING
            </span>
            <p style={{ minHeight: 60, margin: "12px 0 0", color: colors.ink, fontSize: 22, lineHeight: 1.48 }}>
              {response}
              {visibleCharacters < answer.length && (
                <span style={{ color: colors.cyan }}>|</span>
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
        color: colors.ink,
        background:
          "radial-gradient(circle at 18% 15%, rgba(72, 129, 255, .17), transparent 30%), radial-gradient(circle at 82% 80%, rgba(113, 82, 255, .14), transparent 32%), #0b1228",
        fontFamily: '"Manrope Variable", Manrope, "Segoe UI", sans-serif',
      }}
    >
      <div
        style={{
          position: "absolute",
          inset: 0,
          opacity: 0.18,
          backgroundImage:
            "linear-gradient(rgba(145, 178, 255, .16) 1px, transparent 1px), linear-gradient(90deg, rgba(145, 178, 255, .16) 1px, transparent 1px)",
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
        <Img src={brandIcon} style={{ width: 44, height: 44 }} />
        <div>
          <strong style={{ display: "block", fontSize: 21 }}>mycellios</strong>
          <span style={{ color: colors.dim, fontSize: 13, letterSpacing: "0.1em" }}>
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
                  background: "rgba(141, 170, 230, .14)",
                }}
              >
                <span
                  style={{
                    display: "block",
                    width: `${index < activeScene ? 100 : index === activeScene ? sceneProgress : 0}%`,
                    height: "100%",
                    borderRadius: 999,
                    background: index === activeScene ? colors.cyan : colors.blue,
                  }}
                />
              </div>
              <span
                style={{
                  display: "block",
                  marginTop: 8,
                  color: index === activeScene ? colors.ink : colors.dim,
                  fontSize: 12,
                  fontWeight: 750,
                  letterSpacing: "0.1em",
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
          top: 45,
          color: colors.dim,
          fontSize: 14,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {Math.min(20, Math.floor(frame / FPS) + 1)} / {Math.floor(durationInFrames / FPS)} SEC
      </div>
    </AbsoluteFill>
  );
}

export default function ExplainerVideo() {
  return (
    <Player
      component={MycelliosExplainerComposition}
      durationInFrames={DURATION}
      compositionWidth={WIDTH}
      compositionHeight={HEIGHT}
      fps={FPS}
      autoPlay
      loop
      controls
      acknowledgeRemotionLicense
      style={{
        width: "100%",
        aspectRatio: "16 / 9",
        background: "#0b1228",
      }}
    />
  );
}
