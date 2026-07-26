import { describe, expect, it } from "vitest";
import { inferenceMessageWithAttachments, readInferenceAttachment } from "./Panel";

describe("inference attachments", () => {
  it("extracts text files and includes their content in the model message", async () => {
    const file = new File(["Revenue grew 18% in June."], "report.md", { type: "text/markdown" });
    const attachment = await readInferenceAttachment(file, 18_000);

    expect(attachment).toMatchObject({
      name: "report.md",
      kind: "text",
      text: "Revenue grew 18% in June.",
      truncated: false,
    });
    expect(inferenceMessageWithAttachments("Resume el informe.", [attachment])).toContain("Revenue grew 18% in June.");
  });

  it("rejects unsupported binary files instead of pretending the model can read them", async () => {
    const file = new File([new Uint8Array([0, 1, 2])], "archive.zip", { type: "application/zip" });
    await expect(readInferenceAttachment(file, 18_000)).rejects.toThrow("no es compatible");
  });
});
