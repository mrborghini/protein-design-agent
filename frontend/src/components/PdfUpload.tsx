import { useState } from "react";

type PdfInfo = { name: string; pages: number; chars: number };

export default function PdfUpload({ onLoaded }: { onLoaded: (info: PdfInfo) => void }) {
  const [info, setInfo] = useState<PdfInfo | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function upload(file: File) {
    setBusy(true);
    setError("");
    try {
      const body = new FormData();
      body.append("file", file);
      const res = await fetch("/api/upload", { method: "POST", body });
      if (!res.ok) throw new Error((await res.json()).detail ?? "Upload failed");
      const data = (await res.json()) as PdfInfo;
      setInfo(data);
      onLoaded(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <label
        className="flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-300 bg-white px-4 py-6 text-center transition hover:border-slate-400 dark:border-[#4a4a4a] dark:bg-[#3c3c3c] dark:hover:border-slate-500"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          const f = e.dataTransfer.files?.[0];
          if (f) void upload(f);
        }}
      >
        <span className="text-sm font-medium text-slate-700 dark:text-[#ededed]">
          {busy ? "Reading PDF…" : "Drop a PDF or click to upload"}
        </span>
        <span className="mt-1 text-xs text-slate-400 dark:text-[#c8c8c8]">Used as grounding context</span>
        <input
          type="file"
          accept="application/pdf"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void upload(f);
          }}
        />
      </label>

      {info && (
        <div className="mt-3 rounded-lg bg-slate-100 px-3 py-2 text-xs text-slate-600 dark:bg-[#454545] dark:text-[#dcdcdc]">
          <div className="font-medium text-slate-800 dark:text-white">{info.name}</div>
          {info.pages} pages · {info.chars.toLocaleString()} chars
        </div>
      )}
      {error && <p className="mt-2 text-xs text-red-600">{error}</p>}
    </div>
  );
}
