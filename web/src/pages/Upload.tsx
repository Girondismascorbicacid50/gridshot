import { useEffect, useRef, useState } from "react";
import { getHealth, startSession, type BinStyle, type Health } from "../api";
import { useApp } from "../state";
import { InfoTip, ParallaxDiagram } from "../components/InfoTip";

export function Upload() {
  const {
    setTracing,
    setEditor,
    setError,
    error,
  } = useApp();
  const [file1, setFile1] = useState<File | null>(null);
  const [file2, setFile2] = useState<File | null>(null);
  const [prev1, setPrev1] = useState<string | null>(null);
  const [prev2, setPrev2] = useState<string | null>(null);
  const [thickness, setThickness] = useState<number | "">("");
  const [fullHeight, setFullHeight] = useState<number | "">("");
  const [clearance, setClearance] = useState(1.0);
  const [binStyle, setBinStyle] = useState<BinStyle>("pocket");
  const [depth, setDepth] = useState<number | "">("");
  const [overallHeight, setOverallHeight] = useState<number | "">("");
  const [finger, setFinger] = useState(true);
  const [lip, setLip] = useState(true);
  const [health, setHealth] = useState<Health | null>(null);
  const ref1 = useRef<HTMLInputElement>(null);
  const ref2 = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  async function run() {
    if (!file1) return;
    setTracing();
    try {
      const session = await startSession(file1, file2);
      setEditor(session, {
        thickness: thickness === "" ? null : thickness,
        full_height: fullHeight === "" ? null : fullHeight,
        clearance,
        bin_style: binStyle,
        depth: depth === "" ? null : depth,
        overall_height: overallHeight === "" ? null : overallHeight,
        finger_hole: finger,
        lip,
      });
    } catch (e) {
      setError((e as Error).message);
    }
  }

  const matReady = health && health.mats.length > 0;
  // two photos → auto thickness; else a manual thickness is required
  const autoThickness = !!(file1 && file2);
  const canTrace = matReady && file1 && (file2 || thickness !== "");

  return (
    <div className="mx-auto max-w-container px-6 py-12">
      <header className="mb-12">
        <div className="specline border-y border-line py-3 mb-8 flex flex-wrap gap-x-6 gap-y-2">
          <span>Photo → Gridfinity Bin</span>
          <span>Local · GPU</span>
          <span>
            Segserver{" "}
            <span style={{ color: health?.segserver ? "var(--c-teal)" : "var(--c-orange)" }}>
              {health?.segserver ? "online" : "offline"}
            </span>
          </span>
        </div>
        <h1 className="titledev text-4xl leading-none">
          <span className="text-teal">GRID</span>
          <span className="text-muted">•</span>
          <span>SHOT</span>
        </h1>
        <p className="font-body text-lg max-w-[62ch] mt-4">
          Take two photos of a tool on the calibration mat. GridShot uses them
          to determine the tool's true shape and thickness, corrects for camera
          distortion, and creates a print-ready Gridfinity bin.
        </p>
      </header>

      {error && (
        <div className="panel mb-8 border-orange" role="alert">
          <div className="grp-label text-orange-text mb-2">Trace failed</div>
          <p className="font-mono text-sm">{error}</p>
        </div>
      )}

      {!matReady && (
        <div className="panel mb-8" role="status">
          <div className="grp-label mb-2">No verified mat</div>
          <p className="font-body">
            Generate and verify a calibration mat first:{" "}
            <code className="font-mono text-sm">gridshot mat new</code> then{" "}
            <code className="font-mono text-sm">gridshot mat verify</code>.
          </p>
        </div>
      )}


      <div className="grid gap-8 md:grid-cols-2">
        {/* capture — two angles */}
        <div className="panel">
          <div className="grp-label mb-1">01 · Capture</div>
          <p className="font-mono text-xs text-muted mb-4">
            Two angles → thickness solved automatically. One photo → set thickness below.
          </p>
          <div className="grid grid-cols-2 gap-3">
            <Slot
              n={1}
              preview={prev1}
              onPick={(f) => {
                setFile1(f);
                setPrev1(f ? URL.createObjectURL(f) : null);
              }}
              inputRef={ref1}
            />
            <Slot
              n={2}
              optional
              preview={prev2}
              onPick={(f) => {
                setFile2(f);
                setPrev2(f ? URL.createObjectURL(f) : null);
              }}
              inputRef={ref2}
            />
          </div>
          {autoThickness && (
            <div className="mt-3">
              <span className="badge text-teal border-teal">Auto thickness</span>
            </div>
          )}
        </div>

        {/* parameters */}
        <div className="panel">
          <div className="grp-label mb-4">02 · Bin</div>
          <div className="space-y-5">
            <Field
              label="Bin style"
              hint={
                binStyle === "pocket"
                  ? "solid bin with the tool recessed into a fitted pocket"
                  : binStyle === "corral"
                    ? "recessed tool shelf with a full-height separator and stackable perimeter"
                    : "stackable corral with complete Gridfinity sockets wherever they fit outside the tool wall"
              }
            >
              <div className="grid grid-cols-3 gap-2">
                {(["pocket", "corral", "grid"] as BinStyle[]).map((style) => (
                  <button
                    key={style}
                    type="button"
                    aria-pressed={binStyle === style}
                    className={`btn text-xs ${binStyle === style ? "border-teal text-teal" : "btn-ghost"}`}
                    onClick={() => setBinStyle(style)}
                  >
                    {style === "pocket" ? "Pocket" : style === "corral" ? "Stackable corral" : "Live grid"}
                  </button>
                ))}
              </div>
            </Field>
            <Field
              label="Widest-outline height (mm)"
              hint={
                autoThickness
                  ? "measured from your two photos — override only if needed"
                  : "height off the mat at the tool's widest part — NOT its overall height (≈ half for rounded tools). tap ?"
              }
              info={
                <InfoTip label="Which height to measure">
                  <ParallaxDiagram />
                  <p className="font-mono text-xs mb-2 text-teal uppercase" style={{ letterSpacing: "0.12em" }}>
                    Where to measure
                  </p>
                  <p className="font-body text-sm mb-3">
                    The height off the mat at the tool's <strong>widest</strong> part — the
                    part that forms most of the outline — <strong>not</strong> its overall
                    height. It scales the parallax correction, so the wrong height leaves the
                    pocket too big or too small.
                  </p>
                  <ul className="font-body text-sm mb-3 list-disc pl-4 space-y-1">
                    <li><strong>Flat / prismatic</strong> (wrench, gauge): its full thickness.</li>
                    <li><strong>Stepped</strong> (caliper): the long beam (≈ 4 mm), not the tall display.</li>
                    <li><strong>Rounded / domed</strong> (tape measure, mouse): widest in the
                      middle, so ≈ <strong>half</strong> the total height.</li>
                  </ul>
                  <p className="font-mono text-xs mb-2 text-teal uppercase" style={{ letterSpacing: "0.12em" }}>
                    Why
                  </p>
                  <p className="font-body text-sm mb-3">
                    The camera sees slightly over the tool's raised edges (Δ above),
                    stretching the traced outline outward — more for taller edges. This
                    number lets GridShot undo it.
                  </p>
                  <p className="font-mono text-xs text-muted">
                    Can't pinpoint the widest height (any rounded tool)? Add a second photo
                    from a different angle and GridShot measures it exactly.
                  </p>
                </InfoTip>
              }
            >
              <input
                className="mono-input"
                type="number"
                step={0.5}
                min={0}
                value={thickness}
                placeholder={autoThickness ? "auto" : "required (1 photo)"}
                onChange={(e) => setThickness(e.target.value === "" ? "" : Number(e.target.value))}
              />
            </Field>
            <Field
              label="Full tool height (mm)"
              hint="maximum tool height; used only to size the automatic recess. blank keeps the conservative legacy estimate"
            >
              <input
                className="mono-input"
                type="number"
                step={0.5}
                min={0.1}
                value={fullHeight}
                placeholder="optional"
                onChange={(e) => setFullHeight(e.target.value === "" ? "" : Number(e.target.value))}
              />
            </Field>
            <Field label="Clearance (mm)" hint="0.5 tight · 1.0 snug · 1.5 loose">
              <input className="mono-input" type="number" step={0.25} min={0} value={clearance} onChange={(e) => setClearance(Number(e.target.value))} />
            </Field>
            <Field
              label={binStyle === "corral" ? "Tool recess depth (mm)" : "Pocket depth (mm)"}
              hint={binStyle === "corral"
                ? "distance from the stacking plane down to the thin tool shelf. The full-height separator keeps loose parts out. blank = auto"
                : "how deep to recess the tool so you can grip it to lift it out. blank = auto from full tool height"}
            >
              <input className="mono-input" type="number" min={0.1} step={0.5} value={depth} placeholder="auto" onChange={(e) => setDepth(e.target.value === "" ? "" : Number(e.target.value))} />
            </Field>
            <Field label="Overall height (mm)" hint="force a fixed finished height so a set of bins sits level in a drawer. blank = auto (just deep enough for the pocket). snaps to 7mm units; the stacking lip is counted, so lip on/off is handled for you.">
              <input className="mono-input" type="number" step={1} value={overallHeight} placeholder="auto" onChange={(e) => setOverallHeight(e.target.value === "" ? "" : Number(e.target.value))} />
            </Field>
            <div className="flex gap-6">
              <Toggle
                label={binStyle === "corral" ? "Enclosed finger access" : "Finger hole"}
                on={finger}
                set={setFinger}
              />
              <Toggle label="Stacking lip" on={lip} set={setLip} />
            </div>
          </div>
        </div>
      </div>

      <div className="mt-8 flex items-center gap-4">
        <button className="btn btn-primary" disabled={!canTrace} onClick={run}>
          Trace tool
        </button>
        <span className="specline">
          {file1 ? `${file2 ? "2 photos" : "1 photo"} · ${file1.name}` : "no photo selected"}
        </span>
      </div>

    </div>
  );
}

function Slot({
  n,
  optional,
  preview,
  onPick,
  inputRef,
}: {
  n: number;
  optional?: boolean;
  preview: string | null;
  onPick: (f: File | null) => void;
  inputRef: React.RefObject<HTMLInputElement | null>;
}) {
  return (
    <div>
      <button
        className="w-full aspect-square border border-line bg-paper-2 flex items-center justify-center overflow-hidden"
        style={{ borderRadius: 2 }}
        onClick={() => inputRef.current?.click()}
      >
        {preview ? (
          <img src={preview} alt={`angle ${n}`} className="w-full h-full object-cover" />
        ) : (
          <span className="font-mono text-xs text-muted uppercase text-center leading-relaxed" style={{ letterSpacing: "0.1em" }}>
            Angle {n}
            {optional && <><br />(optional)</>}
          </span>
        )}
      </button>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        capture="environment"
        className="hidden"
        onChange={(e) => onPick(e.target.files?.[0] ?? null)}
      />
    </div>
  );
}

function Field({
  label,
  hint,
  info,
  children,
}: {
  label: string;
  hint?: string;
  info?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <label className="block">
      <div className="font-mono text-xs mb-1 text-field flex items-center gap-2">
        <span>{label}</span>
        {info}
      </div>
      {children}
      {hint && <div className="font-mono text-xs text-muted mt-1">{hint}</div>}
    </label>
  );
}

function Toggle({ label, on, set }: { label: string; on: boolean; set: (b: boolean) => void }) {
  return (
    <label className="flex items-center gap-2 cursor-pointer font-mono text-xs uppercase" style={{ letterSpacing: "0.1em" }}>
      <input type="checkbox" checked={on} onChange={(e) => set(e.target.checked)} className="accent-teal w-4 h-4" />
      {label}
    </label>
  );
}
