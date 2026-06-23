// Small browser helpers for file download / upload and friendly formatting.
// No external deps — Blob + URL.createObjectURL + FileReader are all built in.

/** Trigger a browser download of `text` as a file. */
export function downloadText(filename: string, text: string, mime = "application/json"): void {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

/** Download an object as pretty-printed JSON. */
export function downloadJSON(filename: string, data: unknown): void {
  downloadText(filename, JSON.stringify(data, null, 2), "application/json");
}

/** Read a File into a base64 data URL (e.g. "data:image/png;base64,…"). */
export function fileToDataURL(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error ?? new Error("Could not read file"));
    reader.readAsDataURL(file);
  });
}

/** Render a token count as a friendly "K" value, e.g. 32768 → "32K". */
export function formatCtx(n: number): string {
  if (n >= 1024) {
    const k = n / 1024;
    return `${Number.isInteger(k) ? k : k.toFixed(1)}K`;
  }
  return String(n);
}

/** Group separators for token counts, e.g. 12345 → "12,345". */
export function formatTokens(n: number): string {
  return n.toLocaleString();
}
