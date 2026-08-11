import { useEffect, useMemo, useRef, useState } from "react";
import {
  combineLibrary,
  combinePreview,
  combinePreviewGlb,
  type CombinePreview,
  type CombineTool,
  type CombineToolOverride,
  type BinStyle,
  type Placement,
} from "../api";
import { BinViewer } from "./BinViewer";

const PAL = ["#d65a54", "#5ab478", "#548cd6", "#e6be46", "#c85ac8", "#50c8c8", "#e69646", "#a050d6"];

type Pt = [number, number];

/** Apply a placement to a centroid-normalised stamp — rotate CCW about the
 *  origin (matching shapely on the server), then translate. */
function placed(stamp: Pt[], tx: number, ty: number, rot: number): Pt[] {
  const a = (rot * Math.PI) / 180;
  const c = Math.cos(a), s = Math.sin(a);
  return stamp.map(([x, y]) => [x * c - y * s + tx, x * s + y * c + ty]);
}

function placedPoint(point: Pt, tx: number, ty: number, rot: number): Pt {
  return placed([point], tx, ty, rot)[0];
}

function placementsFor(tools: CombineTool[]): Placement[] {
  return tools.map(({ id, tx, ty, rot }) => ({ id, tx, ty, rot }));
}

function overridesFor(tools: CombineTool[]): CombineToolOverride[] {
  return tools.map(({ id, finger_hole_override }) => ({
    id,
    finger_hole: finger_hole_override,
  }));
}

/** Interactive multi-tool-bin editor: auto-packed layout you can drag + rotate,
 *  inspect as the exact generated solid, then export the arrangement as one 3MF. */
export function CombineEditor({
  ids,
  overallHeight,
  lip,
  onClose,
}: {
  ids: string[];
  overallHeight: number | null;
  lip: boolean;
  onClose: () => void;
}) {
  const [meta, setMeta] = useState<CombinePreview | null>(null);
  const [tools, setTools] = useState<CombineTool[]>([]);
  const [sel, setSel] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [view, setView] = useState<"arrange" | "preview">("arrange");
  const [glbUrl, setGlbUrl] = useState<string | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [previewErr, setPreviewErr] = useState<string | null>(null);
  const [binStyle, setBinStyle] = useState<BinStyle>("pocket");
  const svgRef = useRef<SVGSVGElement>(null);
  const drag = useRef<{ id: string; ox: number; oy: number } | null>(null);
  const previewSequence = useRef(0);
  const glbUrlRef = useRef<string | null>(null);

  async function load(
    placements?: Placement[],
    overrides: CombineToolOverride[] = overridesFor(tools),
    style: BinStyle = binStyle,
  ) {
    setBusy(true);
    setErr(null);
    try {
      const p = await combinePreview(
        ids,
        placements ?? null,
        overallHeight,
        lip,
        overrides,
        style,
      );
      setMeta(p);
      setTools(p.tools);
      setSel((current) => current && p.tools.some((tool) => tool.id === current) ? current : null);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  useEffect(() => { load(); /* auto-pack on open */ }, []); // eslint-disable-line

  const idsKey = ids.join("|");
  const geometryKey = useMemo(
    () => JSON.stringify(tools.map((tool) => [
      tool.id,
      tool.tx,
      tool.ty,
      tool.rot,
      tool.finger_hole_override,
    ])),
    [tools],
  );

  // Generate after the arrangement settles. This endpoint calls the same solid
  // builder as 3MF export; no browser-side mesh approximation is involved.
  useEffect(() => {
    if (!meta || tools.length < 2) return;
    const sequence = ++previewSequence.current;
    const placements = placementsFor(tools);
    const overrides = overridesFor(tools);
    setPreviewBusy(true);
    setPreviewErr(null);
    const timer = window.setTimeout(() => {
      combinePreviewGlb(
        ids, placements, overallHeight, lip, overrides, binStyle,
      )
        .then((blob) => {
          if (sequence !== previewSequence.current) return;
          const nextUrl = URL.createObjectURL(blob);
          if (glbUrlRef.current) URL.revokeObjectURL(glbUrlRef.current);
          glbUrlRef.current = nextUrl;
          setGlbUrl(nextUrl);
        })
        .catch((reason) => {
          if (sequence === previewSequence.current) {
            setPreviewErr((reason as Error).message);
          }
        })
        .finally(() => {
          if (sequence === previewSequence.current) setPreviewBusy(false);
        });
    }, 450);
    return () => window.clearTimeout(timer);
  }, [idsKey, geometryKey, overallHeight, lip, binStyle, Boolean(meta)]); // eslint-disable-line

  useEffect(() => () => {
    previewSequence.current += 1;
    if (glbUrlRef.current) URL.revokeObjectURL(glbUrlRef.current);
  }, []);

  // live footprint from the current arrangement (mirrors the server's auto_grid)
  const layout = useMemo(() => {
    if (!meta || !tools.length) return null;
    const polys = tools.map((t) => placed(t.stamp, t.tx, t.ty, t.rot));
    const fingerCircles = tools.flatMap((tool) => tool.finger_holes.map(([x, y, diameter]) => {
      const [cx, cy] = placedPoint([x, y], tool.tx, tool.ty, tool.rot);
      return { toolId: tool.id, cx, cy, radius: diameter / 2 };
    }));
    const xs = [
      ...polys.flat().map((p) => p[0]),
      ...fingerCircles.flatMap((hole) => [hole.cx - hole.radius, hole.cx + hole.radius]),
    ];
    const ys = [
      ...polys.flat().map((p) => p[1]),
      ...fingerCircles.flatMap((hole) => [hole.cy - hole.radius, hole.cy + hole.radius]),
    ];
    const minx = Math.min(...xs), maxx = Math.max(...xs);
    const miny = Math.min(...ys), maxy = Math.max(...ys);
    const { pitch, bin_size, wall } = meta;
    const gx = Math.max(1, Math.ceil((maxx - minx + 2 * wall + (pitch - bin_size)) / pitch));
    const gy = Math.max(1, Math.ceil((maxy - miny + 2 * wall + (pitch - bin_size)) / pitch));
    const ow = pitch * gx - (pitch - bin_size), od = pitch * gy - (pitch - bin_size);
    const cx = (minx + maxx) / 2, cy = (miny + maxy) / 2;
    return { polys, fingerCircles, gx, gy, ow, od, cx, cy };
  }, [tools, meta]);

  function toData(e: React.PointerEvent): Pt {
    const svg = svgRef.current!;
    const p = svg.createSVGPoint();
    p.x = e.clientX; p.y = e.clientY;
    const d = p.matrixTransform(svg.getScreenCTM()!.inverse());
    return [d.x, d.y];
  }
  function down(id: string, e: React.PointerEvent) {
    e.stopPropagation();
    setSel(id);
    const t = tools.find((x) => x.id === id)!;
    const [mx, my] = toData(e);
    drag.current = { id, ox: mx - t.tx, oy: my - t.ty };
    (e.target as Element).setPointerCapture(e.pointerId);
  }
  function move(e: React.PointerEvent) {
    if (!drag.current) return;
    const [mx, my] = toData(e);
    const { id, ox, oy } = drag.current;
    setTools((ts) => ts.map((t) => (t.id === id ? { ...t, tx: mx - ox, ty: my - oy } : t)));
  }
  function rotate(deg: number) {
    if (!sel) return;
    setTools((ts) => ts.map((t) => (t.id === sel ? { ...t, rot: t.rot + deg } : t)));
  }
  function setRotation(deg: number) {
    if (!sel || !Number.isFinite(deg)) return;
    setTools((ts) => ts.map((t) => (t.id === sel ? { ...t, rot: deg } : t)));
  }

  async function setFingerHole(enabled: boolean) {
    if (!sel) return;
    const updated = tools.map((tool) => tool.id === sel ? {
      ...tool,
      finger: enabled,
      finger_hole: enabled,
      finger_hole_override: enabled === tool.finger_hole_inherited ? null : enabled,
    } : tool);
    await load(placementsFor(updated), overridesFor(updated));
  }

  async function exportBin() {
    setBusy(true);
    setErr(null);
    try {
      await combineLibrary(
        ids,
        placementsFor(tools),
        overallHeight,
        lip,
        overridesFor(tools),
        binStyle,
      );
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const color = (i: number) => PAL[i % PAL.length];
  const selectedTool = tools.find((tool) => tool.id === sel) ?? null;
  const displayedRotation = selectedTool
    ? ((((selectedTool.rot + 180) % 360) + 360) % 360) - 180
    : 0;
  const m = 8; // viewport margin (mm)
  const vb = layout
    ? `${layout.cx - layout.ow / 2 - m} ${layout.cy - layout.od / 2 - m} ${layout.ow + 2 * m} ${layout.od + 2 * m}`
    : "0 0 100 100";

  return (
    <div className="panel !p-4 sm:!p-6 w-full max-w-[980px] max-h-[calc(100dvh-2rem)] overflow-auto">
      <div className="grp-label mb-2 flex flex-wrap justify-between gap-2">
        <span>Arrange multi-tool bin</span>
        {layout && (
          <span className="text-muted">
            {layout.gx}×{layout.gy}u · {meta!.overall_height_mm}mm tall · {binStyle}
            {binStyle === "grid" ? ` · ${meta!.available_cells.length} live sockets` : ""}
          </span>
        )}
      </div>
      <p className="font-mono text-[10px] text-muted mb-3">
        Drag a tool to move it · select one and use Rotate · Auto-pack re-solves · live grid adds only complete sockets that fit outside every tool wall.
      </p>

      <div className="grid min-w-0 gap-4 lg:grid-cols-[minmax(0,1fr)_220px]">
        <div className="min-w-0">
          <div className="mb-2 grid grid-cols-2 gap-1">
            <button
              className={`btn text-xs ${view === "arrange" ? "border-teal text-teal" : "btn-ghost"}`}
              onClick={() => setView("arrange")}
            >
              Arrange 2D
            </button>
            <button
              className={`btn text-xs ${view === "preview" ? "border-teal text-teal" : "btn-ghost"}`}
              onClick={() => setView("preview")}
            >
              Preview 3D
            </button>
          </div>
          <div className="border border-line bg-field min-w-0 overflow-hidden" style={{ borderRadius: 2 }}>
            {view === "arrange" ? <svg
            ref={svgRef}
            viewBox={vb}
            className="w-full touch-none"
            style={{ minHeight: 280, maxHeight: "60vh", cursor: drag.current ? "grabbing" : "default" }}
            preserveAspectRatio="xMidYMid meet"
            onPointerMove={move}
            onPointerUp={() => (drag.current = null)}
            onPointerDown={() => setSel(null)}
          >
            {layout && (
              <>
                {/* bin footprint + gridfinity cells */}
                <rect x={layout.cx - layout.ow / 2} y={layout.cy - layout.od / 2}
                  width={layout.ow} height={layout.od} fill="#00000022"
                  stroke="#6b7280" strokeWidth={0.6} rx={2} />
                {Array.from({ length: layout.gx - 1 }, (_, i) => {
                  const x = layout.cx - layout.ow / 2 + (i + 1) * meta!.pitch;
                  return <line key={"v" + i} x1={x} y1={layout.cy - layout.od / 2} x2={x} y2={layout.cy + layout.od / 2} stroke="#3a4046" strokeWidth={0.4} />;
                })}
                {Array.from({ length: layout.gy - 1 }, (_, i) => {
                  const y = layout.cy - layout.od / 2 + (i + 1) * meta!.pitch;
                  return <line key={"h" + i} x1={layout.cx - layout.ow / 2} y1={y} x2={layout.cx + layout.ow / 2} y2={y} stroke="#3a4046" strokeWidth={0.4} />;
                })}
                {binStyle === "grid" && meta!.available_cells.map(([cellX, cellY]) => {
                  const x = layout.cx + cellX - meta!.pitch / 2;
                  const y = layout.cy + cellY - meta!.pitch / 2;
                  return <rect
                    key={`socket-${cellX}-${cellY}`}
                    x={x + 2.5}
                    y={y + 2.5}
                    width={meta!.pitch - 5}
                    height={meta!.pitch - 5}
                    rx={3}
                    fill="#2f8f9522"
                    stroke="#2f8f95"
                    strokeWidth={0.7}
                  />;
                })}
                {/* finger-access scallops are part of the exact cut envelope */}
                {layout.fingerCircles.map((hole, index) => {
                  const toolIndex = tools.findIndex((tool) => tool.id === hole.toolId);
                  return <circle
                    key={`${hole.toolId}-finger-${index}`}
                    cx={hole.cx}
                    cy={hole.cy}
                    r={hole.radius}
                    fill={color(toolIndex) + "2f"}
                    stroke={color(toolIndex)}
                    strokeWidth={0.6}
                    strokeDasharray="2 1"
                  />;
                })}
                {/* cleared pockets */}
                {tools.map((t, i) => (
                  <polygon
                    key={t.id}
                    points={layout.polys[i].map((p) => `${p[0]},${p[1]}`).join(" ")}
                    fill={color(i) + (sel === t.id ? "88" : "55")}
                    stroke={color(i)} strokeWidth={sel === t.id ? 1.2 : 0.7}
                    style={{ cursor: "grab" }}
                    onPointerDown={(e) => down(t.id, e)}
                  />
                ))}
              </>
            )}
            </svg> : (
              <div className="relative h-[clamp(280px,55vh,520px)] w-full">
                {glbUrl && <BinViewer url={glbUrl} />}
                {!glbUrl && !previewErr && (
                  <div className="absolute inset-0 grid place-items-center font-mono text-xs text-muted">
                    Building exact bin preview…
                  </div>
                )}
                {previewErr && (
                  <div className="absolute inset-0 grid place-items-center p-6 text-center font-mono text-xs text-orange">
                    {previewErr}
                  </div>
                )}
                <span className="absolute bottom-2 left-3 font-mono text-[10px] text-line">
                  EXACT EXPORT GEOMETRY{previewBusy ? " · UPDATING" : ""}
                </span>
                <span className="absolute bottom-2 right-3 font-mono text-[10px] text-line">DRAG TO ORBIT</span>
              </div>
            )}
          </div>
        </div>

        <div className="space-y-3 min-w-0">
          <div>
            <span className="font-mono text-[10px] uppercase text-muted">Bin style</span>
            <div className="mt-1 grid grid-cols-3 gap-1">
              {(["pocket", "corral", "grid"] as BinStyle[]).map((style) => (
                <button
                  key={style}
                  type="button"
                  aria-pressed={binStyle === style}
                  className={`btn !px-1 !py-2 text-[10px] ${binStyle === style ? "border-teal text-teal" : "btn-ghost"}`}
                  disabled={busy}
                  onClick={() => {
                    setBinStyle(style);
                    void load(placementsFor(tools), overridesFor(tools), style);
                  }}
                >
                  {style === "pocket" ? "Pocket" : style === "corral" ? "Corral" : "Live grid"}
                </button>
              ))}
            </div>
          </div>
          <label className="block">
            <span className="font-mono text-[10px] uppercase text-muted">Rotation (degrees)</span>
            <div className="mt-1 flex items-center gap-2">
              <input
                className="mono-input min-w-0 flex-1 !px-2 !py-1 !text-sm"
                type="number"
                step={0.1}
                disabled={!selectedTool}
                value={selectedTool ? Number(displayedRotation.toFixed(1)) : ""}
                placeholder="Select a tool"
                onChange={(event) => setRotation(Number(event.target.value))}
              />
              <span className="font-mono text-xs text-muted">°</span>
            </div>
          </label>
          <input
            aria-label="Tool rotation"
            className="w-full accent-teal"
            type="range"
            min={-180}
            max={180}
            step={1}
            disabled={!selectedTool}
            value={displayedRotation}
            onChange={(event) => setRotation(Number(event.target.value))}
          />
          <div className="grid grid-cols-4 gap-1">
            <button className="btn btn-ghost text-[10px] !px-1 !py-2" disabled={!sel} onClick={() => rotate(-15)}>−15°</button>
            <button className="btn btn-ghost text-[10px] !px-1 !py-2" disabled={!sel} onClick={() => rotate(-1)}>−1°</button>
            <button className="btn btn-ghost text-[10px] !px-1 !py-2" disabled={!sel} onClick={() => rotate(1)}>+1°</button>
            <button className="btn btn-ghost text-[10px] !px-1 !py-2" disabled={!sel} onClick={() => rotate(15)}>+15°</button>
          </div>
          {selectedTool ? (
            <div className="border border-line bg-field p-3 font-mono text-[10px]" style={{ borderRadius: 2 }}>
              <div className="mb-2 truncate text-xs text-knockout">
                {selectedTool.label || selectedTool.id.slice(0, 8)}
              </div>
              <dl className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-1 text-muted">
                <dt>Clearance · library</dt>
                <dd className="text-knockout">{selectedTool.clearance_mm} mm</dd>
                <dt>{binStyle === "corral" ? "Tool recess" : "Pocket depth"}</dt>
                <dd className="text-knockout">{selectedTool.depth_mm} mm</dd>
                <dt>Depth source</dt>
                <dd className="text-right text-knockout">{selectedTool.depth_mode}</dd>
              </dl>
              <div className="mt-3 flex items-center justify-between gap-2 border-t border-line pt-3">
                <div className="min-w-0">
                  <div className="text-knockout">Finger access</div>
                  <div className="truncate text-muted">
                    {selectedTool.finger_hole_override === null ? "Inherited from library" : "Override for this bin"}
                  </div>
                </div>
                <button
                  aria-pressed={selectedTool.finger_hole}
                  className={`btn shrink-0 !px-3 !py-1 text-[10px] ${selectedTool.finger_hole ? "border-teal text-teal" : "btn-ghost"}`}
                  disabled={busy}
                  onClick={() => void setFingerHole(!selectedTool.finger_hole)}
                >
                  {selectedTool.finger_hole ? "On" : "Off"}
                </button>
              </div>
              {selectedTool.finger_hole_override !== null && (
                <button
                  className="mt-2 w-full text-left text-teal hover:text-knockout"
                  disabled={busy}
                  onClick={() => void setFingerHole(selectedTool.finger_hole_inherited)}
                >
                  ↩ Use library setting ({selectedTool.finger_hole_inherited ? "on" : "off"})
                </button>
              )}
            </div>
          ) : (
            <div className="border border-line p-3 font-mono text-[10px] text-muted">Select a tool to inspect its effective settings.</div>
          )}
          <div className="max-h-[38vh] overflow-auto space-y-1">
            {tools.map((t, i) => (
              <button
                key={t.id}
                className="w-full border px-2 py-1 text-left font-mono text-[10px]"
                style={{ borderRadius: 2, borderColor: sel === t.id ? color(i) : "var(--c-line)" }}
                onClick={() => setSel(t.id)}
              >
                <span className="flex min-w-0 items-center gap-2">
                  <span className="shrink-0" style={{ width: 8, height: 8, background: color(i), display: "inline-block", borderRadius: 2 }} />
                  <span className="truncate text-knockout">{t.label || t.id.slice(0, 6)}</span>
                </span>
                <span className="mt-1 flex flex-wrap gap-x-2 pl-4 text-muted">
                  <span>Clearance {t.clearance_mm} mm</span>
                  <span>Recess {t.depth_mm} mm</span>
                  <span className={t.finger_hole ? "text-teal" : "text-line"}>Finger {t.finger_hole ? "on" : "off"}</span>
                </span>
              </button>
            ))}
          </div>
          {err && <p className="font-mono text-[10px] text-orange">{err}</p>}
          <button className="btn w-full text-xs" disabled={busy} onClick={() => load(undefined, overridesFor(tools), binStyle)}>↻ Auto-pack</button>
          <button className="btn btn-primary w-full" disabled={busy || !tools.length} onClick={exportBin}>
            ↓ Export bin (3MF)
          </button>
          <button className="btn w-full" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}
