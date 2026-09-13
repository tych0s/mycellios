import mammoth from "mammoth";
import { describe, expect, it } from "vitest";

// Minimal DOCX containing one paragraph. Keep the real ZIP/XML parser in this
// regression: the dependency update must preserve entities in document text.
const DOCUMENT = Buffer.from(
  "UEsDBBQAAAAIADJDLV2sbhJangAAANwAAAATAAAAW0NvbnRlbnRfVHlwZXNdLnhtbF2PsQ7CMBBDf6XKitqrGBhQ24UdGPiBU3JtI5pLlBwF/p4EpA6Mlu1nubu9A6Xq5RZOvZpFwhEg6ZkcpsYH4uyMPjqULOMEAfUdJ4J92x5AexZiqaUw1NBdVorRGqquGOWMjnoFTx8NGK8fLiebTFPV6Vcry73CEBarUaxnWNn8bdZ+HK2mrV9oIXpNKVme3NJsjkPLu4KHoYPvqeEDUEsDBBQAAAAIADJDLV0VPIy/hgAAAK8AAAARAAAAd29yZC9kb2N1bWVudC54bWxFzjEOgzAMBdCrREhlrBFDh0A5RG+QEheQcBw5hrS3L6FDl/dl2fpyn63ncSMMat60hmTzvZpVowVI44zk0pUjhmP3YiGnxygTZBYfhUdMaQkTrdA2zQ3ILaEa+myf7D8lY0EKOjxwx7ChqR3FzkzCWWdTr9qZtrn0UE6KchpPfzXwf3H4AlBLAQIUABQAAAAIADJDLV2sbhJangAAANwAAAATAAAAAAAAAAAAAACAAQAAAABbQ29udGVudF9UeXBlc10ueG1sUEsBAhQAFAAAAAgAMkMtXRU8jL+GAAAArwAAABEAAAAAAAAAAAAAAIABzwAAAHdvcmQvZG9jdW1lbnQueG1sUEsFBgAAAAACAAIAgAAAAIQBAAAAAA==",
  "base64",
);

describe("document parser", () => {
  it("extracts a DOCX paragraph through the patched XML dependency", async () => {
    await expect(mammoth.extractRawText({ buffer: DOCUMENT })).resolves.toMatchObject({
      value: "Revenue & growth < 20%\n\n",
      messages: [],
    });
  });
});
