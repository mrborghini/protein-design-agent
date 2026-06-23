import { formatCtx } from "../lib/download";

export default function ContextControl({
  value,
  min,
  max,
  disabled,
  onChange,
}: {
  value: number;
  min: number;
  max: number;
  disabled: boolean;
  onChange: (n: number) => void;
}) {
  return (
    <div>
      <div className="flex items-center justify-between">
        <label className="text-xs font-medium text-slate-500 dark:text-[#d0d0d0]">
          Default context window
        </label>
        <span className="text-xs font-semibold text-slate-700 dark:text-[#ededed]">{formatCtx(value)}</span>
      </div>
      <input
        type="range"
        value={value}
        min={min}
        max={max}
        step={512}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-2 w-full accent-sky-600 disabled:opacity-50"
      />
      <p className="mt-1 text-[11px] leading-snug text-slate-400 dark:text-[#c8c8c8]">
        Seeds new agents. Each agent can override it in the roster. Range {formatCtx(min)}–{formatCtx(max)}.
      </p>
    </div>
  );
}
