import { useMemo, useRef, useState } from "react";
import type { Poly, Ring } from "../api";
import { useZoomPan } from "./useZoomPan";

type Mode = "move" | "insert" | "delete" | "hole";

function ringPath(ring: Ring): string {
  if (ring.length < 3) return "";
  return `M ${ring.map(([x, y]) => `${x} ${y}`).join(" L ")} Z`;
}

function polyPath(poly: Poly): string {
  return [ringPath(poly.exterior), ...poly.holes.map(ringPath)].join(" ");
}

function changedRing(poly: Poly, ringIndex: number, ring: Ring): Poly {
  if (ringIndex < 0) return { ...poly, exterior: ring };
  return {
    ...poly,
    holes: poly.holes.map((value, index) => (index === ringIndex ? ring : value)),
  };
}

function bounds(poly: Poly) {
  const points = [poly.exterior, ...poly.holes].flat();
  const xs = points.map(([x]) => x);
  const ys = points.map(([, y]) => y);
  const minx = Math.min(...xs);
  const maxx = Math.max(...xs);
  const miny = Math.min(...ys);
  const maxy = Math.max(...ys);
  return { minx, miny, maxx, maxy, width: maxx - minx, height: maxy - miny };
}

export function PhysicalCutoutEditor({
  initial,
  busy,
  onSave,
  onCancel,
}: {
  initial: Poly;
  busy: boolean;
  onSave: (polygon: Poly) => void | Promise<void>;
  onCancel: () => void;
}) {
  const initialRef = useRef(initial);
  const [history, setHistory] = useState<Poly[]>([initial]);
  const [historyIndex, setHistoryIndex] = useState(0);
  const [current, setCurrent] = useState(initial);
  const [mode, setMode] = useState<Mode>("move");
  const [holeDraft, setHoleDraft] = useState<Ring>([]);
  const polygonRef = useRef(current);
  const dragVertex = useRef<{ ring: number; point: number } | null>(null);
  const originalBounds = useMemo(() => bounds(initialRef.current), []);
  const padding = Math.max(originalBounds.width, originalBounds.height) * 0.06 + 2;
  const base = {
    x: originalBounds.minx - padding,
    y: originalBounds.miny - padding,
    w: originalBounds.width + 2 * padding,
    h: originalBounds.height + 2 * padding,
  };
  const zp = useZoomPan(base);
  const extent = Math.max(base.w, base.h);
  const vertexRadius = extent / 125 / zp.zoomFactor;
  const stroke = extent / 450 / zp.zoomFactor;
  const currentBounds = bounds(current);
  const dimensions = [currentBounds.width, currentBounds.height].sort((a, b) => b - a);
  const rings = [current.exterior, ...current.holes];
  const changed = JSON.stringify(current) !== JSON.stringify(initialRef.current);

  function replace(poly: Poly) {
    polygonRef.current = poly;
    setCurrent(poly);
  }

  function commit(poly: Poly) {
    const next = [...history.slice(0, historyIndex + 1), poly];
    setHistory(next);
    setHistoryIndex(next.length - 1);
    replace(poly);
  }

  function step(index: number) {
    setHistoryIndex(index);
    replace(history[index]);
  }

  function toData(clientX: number, clientY: number): [number, number] {
    const svg = zp.svgRef.current!;
    const point = svg.createSVGPoint();
    point.x = clientX;
    point.y = clientY;
    const local = point.matrixTransform(svg.getScreenCTM()!.inverse());
    return [local.x, local.y];
  }

  function down(event: React.PointerEvent<SVGSVGElement>) {
    if (mode === "hole") return;
    zp.panStart(event.clientX, event.clientY);
  }

  function move(event: React.PointerEvent<SVGSVGElement>) {
    if (dragVertex.current) {
      const { ring, point } = dragVertex.current;
      const values = ring < 0
        ? polygonRef.current.exterior
        : polygonRef.current.holes[ring];
      replace(changedRing(
        polygonRef.current,
        ring,
        values.map((value, index) => (
          index === point ? toData(event.clientX, event.clientY) : value
        )),
      ));
      return;
    }
    zp.panMove(event.clientX, event.clientY);
  }

  function up(event: React.PointerEvent<SVGSVGElement>) {
    if (dragVertex.current) {
      dragVertex.current = null;
      commit(polygonRef.current);
      return;
    }
    if (mode === "hole") {
      setHoleDraft((points) => [...points, toData(event.clientX, event.clientY)]);
      return;
    }
    zp.panEnd();
  }

  function vertexDown(ring: number, point: number, event: React.PointerEvent) {
    event.stopPropagation();
    if (mode === "move") {
      dragVertex.current = { ring, point };
      (event.target as Element).setPointerCapture?.(event.pointerId);
      return;
    }
    if (mode !== "delete") return;
    const values = ring < 0 ? current.exterior : current.holes[ring];
    if (ring >= 0 && values.length <= 3) {
      commit({ ...current, holes: current.holes.filter((_, index) => index !== ring) });
    } else if (values.length > 3) {
      commit(changedRing(current, ring, values.filter((_, index) => index !== point)));
    }
  }

  function insert(ring: number, point: number, event: React.PointerEvent) {
    event.stopPropagation();
    const values = ring < 0 ? current.exterior : current.holes[ring];
    const a = values[point];
    const b = values[(point + 1) % values.length];
    commit(changedRing(current, ring, [
      ...values.slice(0, point + 1),
      [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] as [number, number],
      ...values.slice(point + 1),
    ]));
  }

  function finishHole() {
    if (holeDraft.length < 3) return;
    commit({ ...current, holes: [...current.holes, holeDraft] });
    setHoleDraft([]);
    setMode("move");
  }

  function modeButton(value: Mode, label: string) {
    return (
      <button
        className={`font-mono text-[10px] uppercase px-3 py-2 ${mode === value ? "bg-teal text-knockout" : "bg-paper text-muted"}`}
        onClick={() => {
          setMode(value);
          if (value !== "hole") setHoleDraft([]);
        }}
      >
        {label}
      </button>
    );
  }

  return (
    <section className="w-full max-w-[900px] max-h-[calc(100dvh-2rem)] overflow-auto border-2 border-teal bg-paper p-4 sm:p-5">
      <div className="grp-label mb-2">Edit physical cutout</div>
      <p className="font-body text-sm text-muted mb-4 max-w-[76ch]">
        This is the reconstructed footprint used by the layout, before clearance.
        Vertex changes here are final physical dimensions; parallax will not be
        applied to them again.
      </p>
      <div className="flex flex-wrap items-center gap-2 mb-3">
        <div className="inline-flex border border-line">
          {modeButton("move", "Move")}
          {modeButton("insert", "Add vertex")}
          {modeButton("delete", "Delete")}
          {modeButton("hole", "Add opening")}
        </div>
        <div className="flex-1" />
        <span className="font-mono text-[10px] text-muted">
          {dimensions[0].toFixed(2)} × {dimensions[1].toFixed(2)} mm
        </span>
        <button className="btn btn-ghost text-xs px-2 py-1" onClick={() => zp.zoomButton(1.3)}>−</button>
        <button className="btn btn-ghost text-xs px-2 py-1" onClick={() => zp.zoomButton(1 / 1.3)}>＋</button>
        <button className="btn btn-ghost text-xs px-2 py-1" onClick={zp.fit}>Fit</button>
      </div>
      {mode === "hole" && (
        <div className="flex flex-wrap items-center gap-2 mb-3">
          <span className="font-mono text-xs text-muted flex-1 min-w-52">
            Tap 3+ points around a real opening such as a wrench ring. It creates
            a raised island in the pocket; it is not the finger-access scallop.
          </span>
          <button className="btn btn-ghost text-xs px-3 py-2" disabled={holeDraft.length < 3} onClick={finishHole}>
            Finish opening
          </button>
          <button className="btn btn-ghost text-xs px-3 py-2" disabled={!holeDraft.length} onClick={() => setHoleDraft([])}>
            Clear
          </button>
        </div>
      )}
      <div className="border border-line bg-field overflow-hidden" style={{ borderRadius: 2 }}>
        <svg
          ref={zp.svgRef}
          viewBox={zp.viewBox}
          onPointerDown={down}
          onPointerMove={move}
          onPointerUp={up}
          className="block w-full touch-none"
          style={{ maxHeight: "65vh", margin: "0 auto" }}
          preserveAspectRatio="xMidYMid meet"
        >
          <path
            d={polyPath(initialRef.current)}
            fill="none"
            fillRule="evenodd"
            stroke="#888"
            strokeWidth={stroke}
            strokeDasharray={`${extent / 90 / zp.zoomFactor}`}
          />
          <path
            d={polyPath(current)}
            fill="rgba(36,110,114,0.28)"
            fillRule="evenodd"
            stroke="var(--c-teal)"
            strokeWidth={stroke}
          />
          {holeDraft.length > 0 && (
            <polyline
              points={holeDraft.map(([x, y]) => `${x},${y}`).join(" ")}
              fill="rgba(200,83,30,0.2)"
              stroke="var(--c-orange)"
              strokeWidth={stroke}
            />
          )}
          {rings.flatMap((ring, visibleIndex) => {
            const ringIndex = visibleIndex - 1;
            const vertices = ring.map(([x, y], point) => (
              <circle
                key={`v-${ringIndex}-${point}`}
                cx={x}
                cy={y}
                r={vertexRadius}
                fill={mode === "delete" ? "var(--c-orange)" : "var(--c-teal)"}
                onPointerDown={(event) => vertexDown(ringIndex, point, event)}
              />
            ));
            if (mode !== "insert") return vertices;
            return [...vertices, ...ring.map(([x, y], point) => {
              const next = ring[(point + 1) % ring.length];
              return (
                <circle
                  key={`i-${ringIndex}-${point}`}
                  cx={(x + next[0]) / 2}
                  cy={(y + next[1]) / 2}
                  r={vertexRadius * 0.7}
                  fill="#888"
                  onPointerDown={(event) => insert(ringIndex, point, event)}
                />
              );
            })];
          })}
        </svg>
      </div>
      <div className="mt-4 flex flex-wrap items-center gap-3">
        <button className="btn btn-ghost text-xs" disabled={busy || historyIndex === 0} onClick={() => step(historyIndex - 1)}>Undo</button>
        <button className="btn btn-ghost text-xs" disabled={busy || historyIndex === history.length - 1} onClick={() => step(historyIndex + 1)}>Redo</button>
        <button className="btn btn-ghost text-xs" disabled={busy || !changed} onClick={() => commit(initialRef.current)}>Reset</button>
        <div className="flex-1" />
        <button className="btn" disabled={busy} onClick={onCancel}>Cancel</button>
        <button className="btn btn-primary" disabled={busy || !changed || current.exterior.length < 3} onClick={() => onSave(current)}>
          {busy ? "Regenerating…" : "Save cutout and regenerate"}
        </button>
      </div>
    </section>
  );
}
