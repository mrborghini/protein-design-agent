import { useEffect, useState } from "react";

const KEY = "pda-theme";

function initialDark(): boolean {
  const saved = localStorage.getItem(KEY);
  if (saved === "dark") return true;
  if (saved === "light") return false;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
}

export function useDarkMode(): [boolean, () => void] {
  const [dark, setDark] = useState(initialDark);

  useEffect(() => {
    const root = document.documentElement;
    root.classList.toggle("dark", dark);
    localStorage.setItem(KEY, dark ? "dark" : "light");
  }, [dark]);

  return [dark, () => setDark((d) => !d)];
}
