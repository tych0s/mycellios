export interface InferenceAttachmentSummary {
  id: string;
  name: string;
  size: number;
  kind: "pdf" | "docx" | "text";
  truncated: boolean;
}

export interface InferenceAttachment extends InferenceAttachmentSummary {
  text: string;
}

export const MAX_INFERENCE_FILES = 5;
export const MAX_INFERENCE_TOTAL_CHARS = 36_000;
const MAX_INFERENCE_FILE_BYTES = 10 * 1024 * 1024;
const MAX_INFERENCE_FILE_CHARS = 18_000;

export const INFERENCE_FILE_ACCEPT = [
  ".pdf", ".docx", ".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml", ".toml",
  ".html", ".css", ".js", ".jsx", ".ts", ".tsx", ".py", ".java", ".c", ".cpp", ".h",
  ".sql", ".log", ".ini", ".env",
].join(",");

export async function readInferenceAttachment(
  file: File,
  remainingChars: number,
): Promise<InferenceAttachment> {
  if (file.size === 0) throw new Error(`${file.name} is empty.`);
  if (file.size > MAX_INFERENCE_FILE_BYTES) {
    throw new Error(`${file.name} exceeds the ${formatFileSize(MAX_INFERENCE_FILE_BYTES)} limit.`);
  }
  const extension = file.name.split(".").at(-1)?.toLowerCase() ?? "";
  let text = "";
  let kind: InferenceAttachment["kind"] = "text";
  if (extension === "pdf" || file.type === "application/pdf") {
    kind = "pdf";
    const pdfjs = await import("pdfjs-dist");
    pdfjs.GlobalWorkerOptions.workerSrc = new URL(
      "pdfjs-dist/build/pdf.worker.min.mjs",
      import.meta.url,
    ).toString();
    const document = await pdfjs.getDocument({
      data: new Uint8Array(await file.arrayBuffer()),
    }).promise;
    const pages: string[] = [];
    for (let pageNumber = 1; pageNumber <= document.numPages; pageNumber += 1) {
      const page = await document.getPage(pageNumber);
      const content = await page.getTextContent();
      pages.push(
        `[Page ${pageNumber}]\n${content.items.map((item) => "str" in item ? item.str : "").join(" ")}`,
      );
      if (pages.join("\n\n").length >= Math.min(MAX_INFERENCE_FILE_CHARS, remainingChars)) break;
    }
    text = pages.join("\n\n");
  } else if (
    extension === "docx"
    || file.type === "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
  ) {
    kind = "docx";
    const mammoth = await import("mammoth");
    const result = await mammoth.extractRawText({ arrayBuffer: await file.arrayBuffer() });
    text = result.value;
  } else if (file.type.startsWith("text/") || inferenceTextExtension(extension)) {
    text = await file.text();
  } else {
    throw new Error(
      `${file.name} is not supported. Use PDF, DOCX, text, Markdown, CSV, JSON or source files.`,
    );
  }
  const normalized = text.replace(/\u0000/gu, "").replace(/\r\n/gu, "\n").trim();
  if (!normalized) {
    throw new Error(`Could not extract text from ${file.name}. Scanned PDFs need OCR before attaching.`);
  }
  const limit = Math.max(1, Math.min(MAX_INFERENCE_FILE_CHARS, remainingChars));
  const truncated = normalized.length > limit;
  return {
    id: crypto.randomUUID(),
    name: file.name.replace(/[\u0000-\u001f\u007f]/gu, "").slice(0, 180) || "document",
    size: file.size,
    kind,
    truncated,
    text: normalized.slice(0, limit),
  };
}

function inferenceTextExtension(extension: string): boolean {
  return new Set([
    "txt", "md", "csv", "json", "xml", "yaml", "yml", "toml", "html", "css", "js", "jsx",
    "ts", "tsx", "py", "java", "c", "cpp", "h", "sql", "log", "ini", "env",
  ]).has(extension);
}

export function inferenceMessageWithAttachments(
  prompt: string,
  attachments: readonly InferenceAttachment[],
): string {
  if (attachments.length === 0) return prompt;
  const documents = attachments.map((attachment, index) => [
    `--- FILE ${index + 1}: ${attachment.name} (${formatFileSize(attachment.size)})${attachment.truncated ? " · CONTENT TRUNCATED" : ""} ---`,
    attachment.text,
    `--- END OF ${attachment.name} ---`,
  ].join("\n")).join("\n\n");
  return `${prompt}\n\nThe user attached the following documents as reference material. Analyze their contents and clearly distinguish information from the files.\n\n${documents}`;
}

export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(bytes < 10 * 1024 ? 1 : 0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
