// Stable client-side ids for roster agents.
//
// `crypto.randomUUID()` only exists in a *secure context* (HTTPS or localhost).
// When the POC is opened over a plain-HTTP LAN origin (e.g. http://192.168.x.x:8000),
// it's undefined and throws "crypto.randomUUID is not a function". Fall back to
// `crypto.getRandomValues` (widely available, non-secure-context safe), then to a
// Math.random id as a last resort. Ids are local-only (stripped before the backend
// call), so collision resistance just needs to be good enough for React keys.
export function newId(): string {
  const c = globalThis.crypto;
  if (c?.randomUUID) return c.randomUUID();
  if (c?.getRandomValues) {
    const b = c.getRandomValues(new Uint8Array(16));
    b[6] = (b[6] & 0x0f) | 0x40; // version 4
    b[8] = (b[8] & 0x3f) | 0x80; // variant
    const hex = Array.from(b, (x) => x.toString(16).padStart(2, "0"));
    return `${hex.slice(0, 4).join("")}-${hex.slice(4, 6).join("")}-${hex
      .slice(6, 8)
      .join("")}-${hex.slice(8, 10).join("")}-${hex.slice(10, 16).join("")}`;
  }
  return `id-${Math.random().toString(36).slice(2)}${Math.random().toString(36).slice(2)}`;
}
