import type { PhotoOutline } from "../api";
import { useZoomPan } from "./useZoomPan";

/** Enlarged photo of a tool with its outline drawn as a crisp SVG overlay —
 *  zoom/pan to inspect whether the outline hugs the tool, and jump straight to
 *  either editor. Sharper than the card thumbnail (renders the full photo). */
export function PhotoLightbox({
  data,
  label,
  onClose,
  onCutout,
  onRefine,
}: {
  data: PhotoOutline;
  label: string;
  onClose: () => void;
  onCutout: () => void;
  onRefine: () => void;
}) {
  const w = data.width ?? 1000;
  const h = data.height ?? 1000;
  const zp = useZoomPan({ x: 0, y: 0, w, h });
  const pts = data.outline.map((p) => `${p[0]},${p[1]}`).join(" ");

  return (
    <div className="panel !p-4 sm:!p-6 max-h-[calc(100dvh-2rem)] overflow-auto" style={{ width: "min(94vw, 900px)" }}>
      <div className="grp-label mb-2 flex items-center justify-between gap-2">
        <span className="truncate">{label || "tool"} — outline on photo</span>
        <span className="flex items-center gap-1">
          <button className="btn btn-ghost text-xs px-2 py-1" title="zoom out" onClick={() => zp.zoomButton(1.3)}>−</button>
          <span className="font-mono text-[10px] text-muted w-9 text-center">{zp.zoomFactor.toFixed(1)}×</span>
          <button className="btn btn-ghost text-xs px-2 py-1" title="zoom in" onClick={() => zp.zoomButton(1 / 1.3)}>＋</button>
          <button className="btn btn-ghost text-xs px-2 py-1" title="fit" onClick={zp.fit}>Fit</button>
        </span>
      </div>
      <p className="font-mono text-[10px] text-muted mb-2">scroll or ＋/− to zoom · drag to pan</p>
      <div className="border border-line bg-field overflow-hidden" style={{ borderRadius: 2 }}>
        <svg
          ref={zp.svgRef}
          viewBox={zp.viewBox}
          className="block w-full touch-none"
          style={{ maxHeight: "74vh", margin: "0 auto", cursor: "grab" }}
          preserveAspectRatio="xMidYMid meet"
          onPointerDown={(e) => zp.panStart(e.clientX, e.clientY)}
          onPointerMove={(e) => zp.panMove(e.clientX, e.clientY)}
          onPointerUp={() => zp.panEnd()}
        >
          <image href={data.display} width={w} height={h} />
          {data.outline.length > 2 && (
            <>
              <polygon points={pts} fill="none" stroke="#000" strokeWidth={(w / 160) / zp.zoomFactor} />
              <polygon points={pts} fill="rgba(47,199,199,0.10)" stroke="#2fd7d7" strokeWidth={(w / 300) / zp.zoomFactor} />
            </>
          )}
        </svg>
      </div>
      <div className="flex flex-wrap gap-3 mt-3">
        <button className="btn" onClick={onRefine}>◐ Correct photo selection</button>
        <button className="btn" onClick={onCutout}>✎ Edit physical cutout</button>
        <div className="flex-1" />
        <button className="btn btn-primary" onClick={onClose}>Close</button>
      </div>
    </div>
  );
}
