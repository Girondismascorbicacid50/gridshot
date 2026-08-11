import { useRef, useState } from "react";
import type {
  ClickPoint,
  OutlineEditSession,
  OutlineEditState,
  OutlineVariant,
  Poly,
} from "../api";
import { useZoomPan } from "./useZoomPan";

type Box = [number, number, number, number];
type Mode = "box" | "add" | "remove";
type Overlay = "raw" | "clean" | "both";
type EditableState = OutlineEditState & { raw?: Poly; corrected?: Poly };

interface Props {
  session: OutlineEditSession;
  onPrompt: (points: ClickPoint[], box?: Box | null) => Promise<EditableState>;
  onHistory: (direction: "undo" | "redo") => Promise<EditableState>;
  onChange?: (state: EditableState) => void;
  onVariantChange?: (variant: OutlineVariant) => void;
  onSave?: (state: EditableState, variant: OutlineVariant) => void | Promise<void>;
  onCancel?: () => void;
  title?: string;
  saveLabel?: string;
}

function ringPath(ring: [number, number][]): string {
  if (ring.length < 3) return "";
  return `M ${ring.map(([x, y]) => `${x} ${y}`).join(" L ")} Z`;
}

function polyPath(poly: Poly): string {
  return [ringPath(poly.exterior), ...poly.holes.map(ringPath)].join(" ");
}

/** Photo-selection correction shared by capture, batch, and library. This
 * surface changes the segmentation mask only; vertex editing belongs to the
 * separate millimetre-space PhysicalCutoutEditor. */
export function OutlineCorrectionEditor({
  session,
  onPrompt,
  onHistory,
  onChange,
  onVariantChange,
  onSave,
  onCancel,
  title = "Correct outline",
  saveLabel = "Save outline",
}: Props) {
  const [edit, setEdit] = useState<EditableState>(session);
  const [variant, setVariant] = useState<OutlineVariant>(session.accepted_variant);
  const [mode, setMode] = useState<Mode>("box");
  const [overlay, setOverlay] = useState<Overlay>("both");
  const [showPhoto, setShowPhoto] = useState(true);
  const [showMask, setShowMask] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [box, setBox] = useState<Box | null>(null);
  const boxStart = useRef<[number, number] | null>(null);
  const requestSeq = useRef(0);
  const zp = useZoomPan({ x: 0, y: 0, w: session.width, h: session.height });
  const hit = session.width / 55 / zp.zoomFactor;

  function accept(next: EditableState) {
    setEdit(next);
    setVariant(next.accepted_variant);
    onVariantChange?.(next.accepted_variant);
    onChange?.(next);
    if (next.iou_with_previous >= 0.995 && Math.abs(next.area_change_pct) < 0.5) {
      setNotice("The outline barely changed. Try a tighter box or place a point farther inside the missing area.");
    } else {
      setNotice(null);
    }
  }

  async function run(action: () => Promise<EditableState>) {
    const seq = ++requestSeq.current;
    setBusy(true);
    setError(null);
    try {
      const next = await action();
      if (seq !== requestSeq.current) return;
      accept(next);
    } catch (reason) {
      if (seq === requestSeq.current) setError((reason as Error).message);
    } finally {
      if (seq === requestSeq.current) setBusy(false);
    }
  }

  function prompt(points: ClickPoint[], bx?: Box | null) {
    setEdit((current) => ({ ...current, points }));
    return run(() => onPrompt(points, bx));
  }

  function toData(clientX: number, clientY: number): [number, number] {
    const svg = zp.svgRef.current!;
    const point = svg.createSVGPoint();
    point.x = clientX;
    point.y = clientY;
    const local = point.matrixTransform(svg.getScreenCTM()!.inverse());
    return [local.x, local.y];
  }

  function pointPrompt(clientX: number, clientY: number) {
    const [x, y] = toData(clientX, clientY);
    let nearest = -1;
    let distance = hit;
    edit.points.forEach((point, index) => {
      const candidate = Math.hypot(point.x - x, point.y - y);
      if (candidate < distance) {
        nearest = index;
        distance = candidate;
      }
    });
    if (nearest >= 0) prompt(edit.points.filter((_, index) => index !== nearest));
    else prompt([...edit.points, { x, y, label: mode === "remove" ? 0 : 1 }]);
  }

  function down(event: React.PointerEvent<SVGSVGElement>) {
    if (mode === "box") {
      const [x, y] = toData(event.clientX, event.clientY);
      boxStart.current = [x, y];
      setBox([x, y, x, y]);
      event.currentTarget.setPointerCapture(event.pointerId);
      return;
    }
    zp.panStart(event.clientX, event.clientY);
  }

  function move(event: React.PointerEvent<SVGSVGElement>) {
    if (mode === "box" && boxStart.current) {
      const [x, y] = toData(event.clientX, event.clientY);
      const [sx, sy] = boxStart.current;
      setBox([Math.min(sx, x), Math.min(sy, y), Math.max(sx, x), Math.max(sy, y)]);
      return;
    }
    zp.panMove(event.clientX, event.clientY);
  }

  function up(event: React.PointerEvent<SVGSVGElement>) {
    if (mode === "box" && boxStart.current) {
      const [sx, sy] = boxStart.current;
      const [x, y] = toData(event.clientX, event.clientY);
      boxStart.current = null;
      setBox(null);
      const next: Box = [Math.min(sx, x), Math.min(sy, y), Math.max(sx, x), Math.max(sy, y)];
      if (next[2] - next[0] > 15 && next[3] - next[1] > 15) prompt([], next);
      return;
    }
    if (zp.panEnd()) return;
    if (mode === "add" || mode === "remove") pointPrompt(event.clientX, event.clientY);
  }

  function modeButton(value: Mode, label: string) {
    return (
      <button
        className={`font-mono text-[10px] uppercase px-3 py-2 ${mode === value ? "bg-teal text-knockout" : "bg-paper text-muted"}`}
        onClick={() => setMode(value)}
      >
        {label}
      </button>
    );
  }

  const changed =
    edit.history_length > 1 ||
    edit.operation !== "initial" ||
    variant !== session.accepted_variant;

  function chooseVariant(next: OutlineVariant) {
    if (next === "cleaned" && !edit.cleanup.available) return;
    setVariant(next);
    onVariantChange?.(next);
  }

  return (
    <div className="panel !p-4 sm:!p-6 max-h-[calc(100dvh-2rem)] overflow-auto" style={{ maxWidth: 860, width: "94vw" }}>
      <div className="grp-label mb-2 flex justify-between gap-4">
        <span>{title}</span>
        <span className="text-muted">R{edit.revision} · {edit.history_index + 1}/{edit.history_length}{busy ? " · working…" : ""}</span>
      </div>
      <p className="font-body text-sm text-muted mb-3">
        Correct the photographed selection with an AI box or add/remove points.
        Final vertex and opening edits are made on the physical cutout after reconstruction.
      </p>
      <div className="flex flex-wrap gap-2 mb-3">
        <div className="inline-flex border border-line">{modeButton("box", "AI box")}{modeButton("add", "AI add")}{modeButton("remove", "AI remove")}</div>
        <div className="flex-1" />
        <button className="btn btn-ghost text-xs px-2 py-1" onClick={() => zp.zoomButton(1.3)}>−</button>
        <span className="font-mono text-[10px] text-muted self-center">{zp.zoomFactor.toFixed(1)}×</span>
        <button className="btn btn-ghost text-xs px-2 py-1" onClick={() => zp.zoomButton(1 / 1.3)}>＋</button>
        <button className="btn btn-ghost text-xs px-2 py-1" onClick={zp.fit}>Fit</button>
      </div>
      <div className="flex flex-wrap items-center gap-3 mb-3 font-mono text-[10px] text-muted">
        <label><input type="checkbox" checked={showPhoto} onChange={(event) => setShowPhoto(event.target.checked)} /> photo</label>
        <label><input type="checkbox" checked={showMask} onChange={(event) => setShowMask(event.target.checked)} /> mask</label>
        <span>compare</span>
        {(["raw", "clean", "both"] as Overlay[]).map((value) => (
          <button key={value} className={overlay === value ? "text-teal underline" : ""} onClick={() => setOverlay(value)}>{value}</button>
        ))}
        <span className="ml-auto">{edit.diagnostics.vertex_count} vertices · {edit.diagnostics.hole_count} openings · Δ area {edit.diagnostics.area_change_from_initial_pct.toFixed(1)}%</span>
      </div>
      <div className="mb-3 border border-line bg-paper p-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-[10px] uppercase text-muted mr-1">Use for cutout</span>
          <button
            className={`btn text-xs ${variant === "cleaned" ? "btn-primary" : "btn-ghost"}`}
            disabled={!edit.cleanup.available}
            onClick={() => chooseVariant("cleaned")}
          >
            Cleaned{edit.cleanup.recommended === "cleaned" ? " · recommended" : ""}
          </button>
          <button
            className={`btn text-xs ${variant === "raw" ? "btn-primary" : "btn-ghost"}`}
            onClick={() => chooseVariant("raw")}
          >
            Raw segmentation{edit.cleanup.recommended === "raw" ? " · recommended" : ""}
          </button>
        </div>
        <p className="font-mono text-[10px] text-muted mt-2">
          {edit.cleanup.available
            ? `Noise ${edit.cleanup.noise_mm.toFixed(2)}mm · cleanup radius ${edit.cleanup.radius_mm.toFixed(2)}mm · maximum shift ${edit.cleanup.max_shift_mm.toFixed(2)}mm of ${edit.cleanup.max_shift_cap_mm.toFixed(2)}mm allowed${edit.cleanup.straightened ? " · straight edges detected" : ""}`
            : edit.cleanup.reason ?? "A bounded cleanup candidate is unavailable; raw is preserved."}
        </p>
      </div>
      {error && <p className="font-mono text-xs text-orange mb-3">Edit failed: {error}</p>}
      {notice && <p className="font-mono text-xs text-muted mb-3">{notice}</p>}
      <div className="border border-line bg-field overflow-hidden" style={{ borderRadius: 2 }}>
        <svg
          ref={zp.svgRef}
          viewBox={zp.viewBox}
          onPointerDown={down}
          onPointerMove={move}
          onPointerUp={up}
          className="block w-full touch-none"
          style={{ maxHeight: "65vh", margin: "0 auto", cursor: mode === "box" ? "crosshair" : "default" }}
          preserveAspectRatio="xMidYMid meet"
        >
          {showPhoto && <image href={session.display} width={session.width} height={session.height} />}
          {showMask && overlay !== "clean" && (
            <path d={polyPath(edit.polygon)} fill="rgba(36,110,114,0.28)" fillRule="evenodd" stroke="var(--c-teal)" strokeWidth={session.width / 500 / zp.zoomFactor} />
          )}
          {showMask && overlay !== "raw" && (
            <path d={polyPath(edit.cleaned_polygon)} fill={overlay === "clean" ? "rgba(230,190,70,0.18)" : "none"} fillRule="evenodd" stroke="#e6be46" strokeWidth={session.width / 650 / zp.zoomFactor} strokeDasharray={`${session.width / 180 / zp.zoomFactor}`} />
          )}
          {box && <rect x={box[0]} y={box[1]} width={box[2] - box[0]} height={box[3] - box[1]} fill="rgba(230,190,70,0.15)" stroke="#e6be46" strokeWidth={session.width / 400 / zp.zoomFactor} />}
          {(mode === "add" || mode === "remove") && edit.points.map((point, index) => (
            <circle key={`prompt-${index}`} cx={point.x} cy={point.y} r={hit / 2} fill={point.label ? "var(--c-teal)" : "var(--c-orange)"} stroke="var(--c-knockout)" strokeWidth={session.width / 900 / zp.zoomFactor} />
          ))}
        </svg>
      </div>
      <div className="mt-4 flex flex-wrap items-center gap-3">
        <button className="btn btn-ghost text-xs" disabled={busy || !edit.can_undo} onClick={() => run(() => onHistory("undo"))}>Undo</button>
        <button className="btn btn-ghost text-xs" disabled={busy || !edit.can_redo} onClick={() => run(() => onHistory("redo"))}>Redo</button>
        <button className="btn btn-ghost text-xs" disabled={busy} onClick={() => prompt([], null)}>Reset</button>
        <div className="flex-1" />
        {onCancel && <button className="btn" onClick={onCancel}>Cancel</button>}
        {onSave && <button className="btn btn-primary" disabled={busy || !changed || edit.polygon.exterior.length < 3} onClick={() => onSave(edit, variant)}>{saveLabel}</button>}
      </div>
    </div>
  );
}
