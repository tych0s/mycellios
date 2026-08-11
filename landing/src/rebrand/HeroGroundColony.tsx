const specimens = [
  { kind: "bell", className: "rb-ground-mushroom-a" },
  { kind: "button", className: "rb-ground-mushroom-b" },
  { kind: "flat", className: "rb-ground-mushroom-c" },
  { kind: "parasol", className: "rb-ground-mushroom-d" },
  { kind: "ink", className: "rb-ground-mushroom-e" },
] as const;

type SpecimenKind = (typeof specimens)[number]["kind"];

function Specimen({ kind, index }: { kind: SpecimenKind; index: number }) {
  const skin = `rb-colony-skin-${index}`;
  const gills = `rb-colony-gills-${index}`;
  return (
    <svg viewBox="0 0 100 140" aria-hidden="true">
      <defs>
        <linearGradient id={skin} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#eee8dc" />
          <stop offset=".55" stopColor="#cbb8a1" />
          <stop offset="1" stopColor="#806044" />
        </linearGradient>
        <linearGradient id={gills} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#d6a66d" />
          <stop offset="1" stopColor="#5d3a27" />
        </linearGradient>
      </defs>
      {kind === "bell" && <>
        <path className="rb-mini-stem" d="M43 132 C44 109 45 81 49 51 C51 42 57 43 59 53 C61 81 59 109 62 132Z" fill={`url(#${skin})`} />
        <path className="rb-mini-cap" d="M22 58 C27 30 39 13 53 12 C67 14 77 31 80 58 C66 63 36 63 22 58Z" fill={`url(#${skin})`} />
        <path className="rb-mini-gills" d="M22 58 Q52 48 80 58 Q53 70 22 58Z" fill={`url(#${gills})`} />
      </>}
      {kind === "button" && <>
        <path className="rb-mini-stem" d="M40 132 C43 106 44 83 47 66 C49 58 57 58 60 67 C62 87 60 110 64 132Z" fill={`url(#${skin})`} />
        <path className="rb-mini-cap" d="M17 68 C23 42 39 29 54 29 C70 29 84 43 88 68 C69 75 36 75 17 68Z" fill={`url(#${skin})`} />
        <path className="rb-mini-gills" d="M17 68 Q53 59 88 68 Q54 79 17 68Z" fill={`url(#${gills})`} />
      </>}
      {kind === "flat" && <>
        <path className="rb-mini-stem" d="M45 132 C44 105 47 76 49 55 C50 48 56 48 58 55 C60 79 58 107 61 132Z" fill={`url(#${skin})`} />
        <path className="rb-mini-cap" d="M8 57 C25 39 75 38 93 57 C78 66 24 66 8 57Z" fill={`url(#${skin})`} />
        <path className="rb-mini-gills" d="M8 57 Q50 51 93 57 Q52 71 8 57Z" fill={`url(#${gills})`} />
      </>}
      {kind === "parasol" && <>
        <path className="rb-mini-stem" d="M47 132 C48 101 47 64 50 39 C51 31 57 31 59 40 C61 69 57 104 61 132Z" fill={`url(#${skin})`} />
        <path className="rb-mini-cap" d="M12 48 C24 27 41 18 55 18 C70 18 84 30 91 48 C72 54 31 54 12 48Z" fill={`url(#${skin})`} />
        <path className="rb-mini-gills" d="M12 48 Q53 39 91 48 Q55 59 12 48Z" fill={`url(#${gills})`} />
      </>}
      {kind === "ink" && <>
        <path className="rb-mini-stem" d="M45 132 C47 101 47 69 50 48 C51 40 57 40 59 49 C60 75 57 105 61 132Z" fill={`url(#${skin})`} />
        <path className="rb-mini-cap" d="M28 58 C31 30 42 11 54 10 C67 12 76 31 79 58 C67 54 39 54 28 58Z" fill={`url(#${skin})`} />
        <path className="rb-mini-gills" d="M28 58 Q53 47 79 58 Q54 67 28 58Z" fill={`url(#${gills})`} />
      </>}
    </svg>
  );
}

export function HeroGroundColony() {
  return (
    <div className="rb-ground-colony" aria-hidden="true">
      {specimens.map((specimen, index) => (
        <i className={`rb-ground-mushroom ${specimen.className}`} key={specimen.kind}>
          <Specimen kind={specimen.kind} index={index} />
        </i>
      ))}
    </div>
  );
}
