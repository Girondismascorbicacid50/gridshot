import { useEffect, useRef, useState } from "react";

/** Tap/click info popover (works on touch — no hover dependency), 2046 style. */
export function InfoTip({ label, children }: { label: string; children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (wrap.current && !wrap.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <span ref={wrap} className="relative inline-block align-middle">
      <button
        type="button"
        aria-label={label}
        aria-expanded={open}
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setOpen((o) => !o);
        }}
        className="font-mono text-xs w-4 h-4 inline-flex items-center justify-center
          border border-line text-muted hover:text-teal hover:border-teal"
        style={{ borderRadius: 9999, lineHeight: 1 }}
      >
        ?
      </button>
      {open && (
        <div
          role="dialog"
          aria-label={label}
          className="absolute left-0 top-6 z-20 w-72 panel p-4 bg-paper-2 shadow-[4px_4px_0_rgba(23,25,28,0.12)]"
        >
          {children}
        </div>
      )}
    </span>
  );
}

/** Side-view parallax diagram: the camera "sees over" a raised edge, so the
 *  traced outline lands outward of the true edge by Δ ∝ height. */
export function ParallaxDiagram() {
  // camera (30,14) → ray through tool top-outer corner (95,64) → mat y=104
  return (
    <svg viewBox="0 0 170 118" className="w-full mb-3" role="img" aria-label="Parallax diagram">
      {/* mat */}
      <line x1="8" y1="104" x2="162" y2="104" stroke="var(--c-field)" strokeWidth="1.5" />
      {/* tool block, height t */}
      <rect x="60" y="64" width="35" height="40" fill="var(--c-teal)" opacity="0.85" />
      <text x="77" y="88" textAnchor="middle" className="font-mono" fontSize="10" fill="var(--c-knockout)">t</text>
      {/* camera */}
      <circle cx="30" cy="14" r="4" fill="var(--c-field)" />
      <text x="30" y="30" textAnchor="middle" className="font-mono" fontSize="8" fill="var(--c-muted)">CAMERA</text>
      {/* sight ray grazing the top-outer edge, landing outward */}
      <line x1="30" y1="14" x2="132" y2="104" stroke="var(--c-orange)" strokeWidth="1.2" strokeDasharray="3 2" />
      {/* true edge vs traced edge */}
      <line x1="95" y1="104" x2="95" y2="110" stroke="var(--c-field)" strokeWidth="1.5" />
      <line x1="132" y1="104" x2="132" y2="110" stroke="var(--c-orange)" strokeWidth="1.5" />
      <text x="113" y="116" textAnchor="middle" className="font-mono" fontSize="8" fill="var(--c-orange)">Δ</text>
    </svg>
  );
}
