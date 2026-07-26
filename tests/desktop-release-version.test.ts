import { describe, expect, it } from "vitest";
import {
  compareVersions,
  currentVersion,
  highestVersion,
  nextPatch,
  parseVersion,
  publishedVersions,
  resolveNextVersion,
} from "../scripts/desktop-release-version.mjs";

describe("desktop release version", () => {
  // LA TRAMPA. Comparar versiones como texto acierta por casualidad mientras
  // los numeros tienen los mismos digitos, y falla en cuanto no los tienen:
  // "0.2.9" > "0.2.10" en orden alfabetico, porque se decide en '9' contra '1'.
  // Un guardian que comparase cadenas daria por buena una version 0.2.9 con el
  // tag v0.2.10 ya publicado, y el pipeline volveria a atascarse.
  it("compares numbers numerically, where text comparison breaks", () => {
    expect("0.2.9" > "0.2.10").toBe(true); // lo que dice el orden alfabetico
    expect(compareVersions("0.2.9", "0.2.10")).toBeLessThan(0); // lo correcto

    expect("0.10.0" > "0.9.9").toBe(false); // idem, y tambien al reves
    expect(compareVersions("0.10.0", "0.9.9")).toBeGreaterThan(0);

    // El caso real del atasco: aqui el texto acierta, pero solo por suerte.
    expect(compareVersions("0.2.19", "0.2.51")).toBeLessThan(0);
    expect(compareVersions("1.0.0", "0.99.99")).toBeGreaterThan(0);
    expect(compareVersions("0.2.52", "0.2.52")).toBe(0);
  });

  it("rejects anything that is not canonical MAJOR.MINOR.PATCH", () => {
    expect(() => parseVersion("v0.2.52")).toThrow("not canonical");
    expect(() => parseVersion("0.2")).toThrow("not canonical");
    expect(() => parseVersion("0.2.52-beta")).toThrow("not canonical");
    expect(() => parseVersion("")).toThrow("not canonical");
    expect(parseVersion(" 0.2.52 ")).toEqual([0, 2, 52]);
  });

  it("picks the highest version rather than the last one seen", () => {
    expect(highestVersion(["0.2.9", "0.2.51", "0.2.19"])).toBe("0.2.51");
    expect(highestVersion(["0.2.51"])).toBe("0.2.51");
    expect(highestVersion([])).toBeNull();
  });

  // El caso exacto del atasco: la version comprometida (0.2.19) esta MUY por
  // detras de lo publicado (0.2.51). La siguiente libre tiene que salir del
  // techo real, no de la version comprometida.
  it("steps past the highest published tag, not past the committed version", () => {
    const publicadas = ["0.2.19", "0.2.44", "0.2.51"];
    expect(resolveNextVersion("0.2.19", publicadas)).toBe("0.2.52");
  });

  it("still steps forward when the committed version leads", () => {
    expect(resolveNextVersion("0.3.0", ["0.2.51"])).toBe("0.3.1");
  });

  it("steps forward with no published history at all", () => {
    expect(resolveNextVersion("0.2.19", [])).toBe("0.2.20");
    expect(nextPatch("0.2.51")).toBe("0.2.52");
  });

  // El invariante que este guion existe para sostener: lo que hay comprometido
  // en el arbol tiene que ir por delante de TODO tag publicado. Si este test
  // falla, el pipeline de publicacion esta atascado y hay que correr
  // `npm run version:bump`.
  it("keeps the committed version ahead of every published tag", () => {
    const declarada = currentVersion();
    const publicadas = publishedVersions();
    if (publicadas === null || publicadas.length === 0) return; // sin git o sin tags
    const tomadas = publicadas.filter(
      (candidata) => compareVersions(candidata, declarada) >= 0,
    );
    expect(
      tomadas,
      `la version comprometida ${declarada} no va por delante de ${tomadas.join(", ")}; `
        + "corre npm run version:bump",
    ).toEqual([]);
  });

  // package.json y package-lock.json tienen que moverse juntos: el propio
  // workflow aborta si difieren, y el lock guarda la version en DOS sitios.
  it("keeps package.json and both package-lock fields in step", () => {
    expect(() => currentVersion()).not.toThrow();
  });
});
