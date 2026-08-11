import { useMemo, useRef, useState } from "react";
import { useZoomPan } from "./useZoomPan";

type Pt = [number, number];

/** Manual outline editor (Tracefinity-style): drag vertices, add on edges,
 *  delete vertices, or Chaikin-smooth. Works in the polygon's own coordinate
 *  space (SVG viewBox = data bounds), so it's frame-agnostic. Touch + mouse. */
export function PolygonEditor({
  initial,
  onSave,
  onCancel,
  image,
}: {
  initial: Pt[];
  onSave: (pts: Pt[]) => void;
  onCancel: () => void;
  image?: { href: string; width: number; height: number };
}) {
  // Photo-backed flows are simplified by the server with a physical error
  // bound. Legacy geometry-only entries have no pixel/mm transform, so retain
  // their source vertices instead of silently reducing them to a fixed count.
  const [pts, setPts] = useState<Pt[]>(() => initial.map((p) => [...p] as Pt));
  const [mode, setMode] = useState<"move" | "add" | "delete">("move");
  const drag = useRef<number | null>(null);

  const vb = useMemo(() => {
    if (image) {
      // fixed to the photo bounds so the whole image stays framed as you edit
      const pad = Math.max(image.width, image.height) * 0.02;
      return { x: -pad, y: -pad, w: image.width + 2 * pad, h: image.height + 2 * pad };
    }
    const xs = pts.map((p) => p[0]);
    const ys = pts.map((p) => p[1]);
    const minx = Math.min(...xs), miny = Math.min(...ys);
    const maxx = Math.max(...xs), maxy = Math.max(...ys);
    const pad = Math.max(maxx - minx, maxy - miny) * 0.08 + 4;
    return { x: minx - pad, y: miny - pad, w: maxx - minx + 2 * pad, h: maxy - miny + 2 * pad };
  }, [pts, image]);
  const zp = useZoomPan(vb);
  const r = Math.max(vb.w, vb.h) / 100 / zp.zoomFactor; // constant on-screen size

  function toData(e: React.PointerEvent): Pt {
    const svg = zp.svgRef.current!;
    const p = svg.createSVGPoint();
    p.x = e.clientX;
    p.y = e.clientY;
    const d = p.matrixTransform(svg.getScreenCTM()!.inverse());
    return [d.x, d.y];
  }
  function vertexDown(i: number, e: React.PointerEvent) {
    e.stopPropagation();
    if (mode === "delete") {
      if (pts.length > 4) setPts(pts.filter((_, k) => k !== i));
    } else if (mode === "move") {
      drag.current = i;
      (e.target as Element).setPointerCapture(e.pointerId);
    }
  }
  function pointerMove(e: React.PointerEvent) {
    if (drag.current !== null) {
      const d = toData(e);
      setPts(pts.map((p, k) => (k === drag.current ? d : p)));
      return;
    }
    zp.panMove(e.clientX, e.clientY); // drag on empty space = pan
  }
  function insertAt(i: number) {
    const a = pts[i], b = pts[(i + 1) % pts.length];
    setPts([...pts.slice(0, i + 1), [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], ...pts.slice(i + 1)]);
  }
  function chaikin() {
    const out: Pt[] = [];
    for (let i = 0; i < pts.length; i++) {
      const a = pts[i], b = pts[(i + 1) % pts.length];
      out.push([0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]]);
      out.push([0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]]);
    }
    setPts(out);
  }

  const path = "M " + pts.map((p) => `${p[0]} ${p[1]}`).join(" L ") + " Z";
  const btn = (m: typeof mode, label: string) => (
    <button
      className={`px-3 py-1 font-mono text-xs uppercase ${mode === m ? "bg-teal text-field" : "text-muted border border-line"}`}
      style={{ letterSpacing: "0.08em", borderRadius: 2 }}
      onClick={() => setMode(m)}
    >
      {label}
    </button>
  );

  return (
    <div className="panel" style={{ maxWidth: 640 }}>
      <div className="grp-label mb-3">Adjust outline</div>
      <div className="flex flex-wrap gap-2 mb-3">
        {btn("move", "Move")}
        {btn("add", "Add")}
        {btn("delete", "Delete")}
        <button className="px-3 py-1 font-mono text-xs uppercase text-muted border border-line" style={{ borderRadius: 2 }} onClick={chaikin}>
          Smooth
        </button>
        <div className="flex-1" />
        <button className="px-2 py-1 font-mono text-xs text-muted border border-line" style={{ borderRadius: 2 }} title="zoom out" onClick={() => zp.zoomButton(1.3)}>−</button>
        <span className="font-mono text-[10px] text-muted self-center w-9 text-center">{zp.zoomFactor.toFixed(1)}×</span>
        <button className="px-2 py-1 font-mono text-xs text-muted border border-line" style={{ borderRadius: 2 }} title="zoom in" onClick={() => zp.zoomButton(1 / 1.3)}>＋</button>
        <button className="px-2 py-1 font-mono text-xs text-muted border border-line" style={{ borderRadius: 2 }} title="fit" onClick={zp.fit}>Fit</button>
      </div>
      <p className="font-mono text-[10px] text-muted mb-2">
        {mode === "move" ? "drag a point to move it" : mode === "add" ? "tap a grey dot on an edge to add a point" : "tap a point to delete it"}
        {" · scroll or ＋/− to zoom · drag empty space to pan"}
      </p>
      <svg
        ref={zp.svgRef}
        viewBox={zp.viewBox}
        className="w-full border border-line bg-field touch-none"
        style={{ borderRadius: 2, maxHeight: "60vh" }}
        onPointerDown={(e) => zp.panStart(e.clientX, e.clientY)}
        onPointerMove={pointerMove}
        onPointerUp={() => { drag.current = null; zp.panEnd(); }}
      >
        {image && <image href={image.href} x={0} y={0} width={image.width} height={image.height} />}
        <path d={path} fill={image ? "#2f8f9533" : "#2f8f9522"} stroke="#2f8f95" strokeWidth={r * 0.4} />
        {mode === "add" &&
          pts.map((p, i) => {
            const b = pts[(i + 1) % pts.length];
            return (
              <circle key={"m" + i} cx={(p[0] + b[0]) / 2} cy={(p[1] + b[1]) / 2} r={r * 0.7}
                fill="#888" style={{ cursor: "copy" }} onPointerDown={(e) => { e.stopPropagation(); insertAt(i); }} />
            );
          })}
        {pts.map((p, i) => (
          <circle key={i} cx={p[0]} cy={p[1]} r={r}
            fill={mode === "delete" ? "#d65a54" : "#2f8f95"}
            style={{ cursor: mode === "delete" ? "pointer" : "move" }}
            onPointerDown={(e) => vertexDown(i, e)} />
        ))}
      </svg>
      <div className="flex gap-3 mt-4">
        <button className="btn btn-primary" onClick={() => onSave(pts)}>Save outline</button>
        <button className="btn" onClick={onCancel}>Cancel</button>
      </div>
    </div>
  );
}
