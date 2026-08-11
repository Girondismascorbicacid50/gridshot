import { useState } from "react";
import { useApp } from "../state";
import {
  addToLibrary,
  getSession,
  sessionGenerate,
  sessionSetPhysicalOutline,
  type BinStyle,
  type GenerateParams,
  type Poly,
} from "../api";
import { Silhouette } from "../components/Silhouette";
import { BinViewer } from "../components/BinViewer";
import { ReadinessPanel } from "../components/ReadinessPanel";
import { PhysicalCutoutEditor } from "../components/PhysicalCutoutEditor";

const DOWNLOAD_ORDER = ["3mf", "stl", "layout", "svg"];

export function Result() {
  const {
    result,
    session,
    params,
    reset,
    setLibrary,
    setEditor,
    setResult,
  } = useApp();
  const [saved, setSaved] = useState(false);
  const [pocketDepth, setPocketDepth] = useState(
    result?.bin.pocket_depth_override_mm?.toString() ?? "",
  );
  const [binStyle, setBinStyle] = useState<BinStyle>(
    result?.bin.bin_style ?? "pocket",
  );
  const [finishedHeight, setFinishedHeight] = useState(
    result?.bin.overall_height_override_mm?.toString() ?? "",
  );
  const [fullToolHeight, setFullToolHeight] = useState(
    result?.bin.full_height_mm?.toString() ?? "",
  );
  const [regenerating, setRegenerating] = useState(false);
  const [returning, setReturning] = useState(false);
  const [editingCutout, setEditingCutout] = useState(false);
  const [controlError, setControlError] = useState<string | null>(null);
  if (!result) return null;
  const {
    bin,
    calibration,
    tool_poly,
    pocket_poly,
    reconstruction,
    warnings,
    readiness,
    files,
  } = result;
  const downloads = DOWNLOAD_ORDER.filter((k) => files[k]);
  const hasLiveSession = Boolean(session && params);
  const artifactUrl = (url: string) =>
    `${url}${url.includes("?") ? "&" : "?"}v=${encodeURIComponent(
      bin.derivation_key,
    )}`;

  async function saveToLibrary() {
    try {
      await addToLibrary(result!.project);
      setSaved(true);
    } catch {
      /* ignore */
    }
  }
  async function regenerate() {
    if (!session || !params) return;
    setRegenerating(true);
    setControlError(null);
    try {
      const nextParams: GenerateParams = {
        ...params,
        bin_style: binStyle,
        depth: parseOptionalPositive(pocketDepth, "Tool recess depth"),
        full_height: parseOptionalPositive(fullToolHeight, "Full tool height"),
        overall_height: parseOptionalPositive(
          finishedHeight,
          "Finished bin height",
        ),
      };
      const nextResult = await sessionGenerate(session.session, nextParams);
      setResult(nextResult, nextParams);
      setSaved(false);
    } catch (reason) {
      setControlError((reason as Error).message);
    } finally {
      setRegenerating(false);
    }
  }

  async function savePhysicalCutout(polygon: Poly) {
    if (!session || !params) return;
    setRegenerating(true);
    setControlError(null);
    try {
      await sessionSetPhysicalOutline(session.session, polygon);
      const nextParams: GenerateParams = {
        ...params,
        bin_style: binStyle,
        depth: parseOptionalPositive(pocketDepth, "Tool recess depth"),
        full_height: parseOptionalPositive(fullToolHeight, "Full tool height"),
        overall_height: parseOptionalPositive(
          finishedHeight,
          "Finished bin height",
        ),
      };
      const nextResult = await sessionGenerate(session.session, nextParams);
      setEditingCutout(false);
      setResult(nextResult, nextParams);
      setSaved(false);
    } catch (reason) {
      setControlError((reason as Error).message);
    } finally {
      setRegenerating(false);
    }
  }

  async function backToEditor() {
    if (!session || !params) return;
    setReturning(true);
    setControlError(null);
    try {
      const latest = await getSession(session.session);
      setEditor(latest, params);
    } catch (reason) {
      setControlError((reason as Error).message);
      setReturning(false);
    }
  }

  const parallax =
    calibration.camera_height_mm && bin.thickness_mm
      ? 1 - bin.thickness_mm / calibration.camera_height_mm
      : null;
  const manualPhysical = reconstruction?.method === "manual_physical_outline";

  return (
    <div className="mx-auto max-w-container px-6 py-12">
      {/* hero: the tool as knockout silhouette + the printable bin in 3D */}
      <div className="grid gap-4 mb-8 md:grid-cols-2">
        <div
          className="w-full border border-line overflow-hidden"
          style={{ aspectRatio: "16/9", borderRadius: 2 }}
        >
          {tool_poly && <Silhouette tool={tool_poly} pocket={pocket_poly} />}
        </div>
        <div
          className="w-full border border-line overflow-hidden relative"
          style={{ aspectRatio: "16/9", borderRadius: 2 }}
        >
          {files.glb && <BinViewer url={artifactUrl(files.glb)} />}
          <span className="absolute bottom-2 right-3 font-mono text-xs text-line" style={{ letterSpacing: "0.12em" }}>
            DRAG TO ORBIT
          </span>
        </div>
      </div>

      <ReadinessPanel readiness={readiness} className="mb-8" />

      {editingCutout && tool_poly && (
        <div className="mt-8">
          <PhysicalCutoutEditor
            initial={physicalEditorPolygon(tool_poly)}
            busy={regenerating}
            onSave={savePhysicalCutout}
            onCancel={() => setEditingCutout(false)}
          />
        </div>
      )}

      {/* spec band */}
      <div className="border-t-2 border-field pt-6">
        <h1 className="titledev text-3xl mb-6">
          <span className="text-teal">TOOL</span>
          <span className="text-muted">•</span>
          <span>CUTOUT</span>
        </h1>

        <div className="grid gap-8 md:grid-cols-3">
          <div>
            <div className="grp-label mb-3">Bin</div>
            <table className="dtable">
              <tbody>
                <Row k="Grid" v={`${bin.grid[0]} × ${bin.grid[1]} u`} />
                <Row
                  k="Style"
                  v={bin.bin_style === "pocket" ? "Pocket" : bin.bin_style === "corral" ? "Stackable corral" : "Live grid"}
                />
                <Row k="Height" v={`${bin.height_u}u${bin.lip ? " + lip" : ""}`} />
                <Row k="Overall height" v={`${bin.overall_height_mm} mm`} />
                <Row
                  k={bin.bin_style === "pocket" ? "Pocket depth" : "Tool recess"}
                  v={`${bin.pocket_depth_mm} mm`}
                />
                {bin.bin_style === "grid" && (
                  <Row
                    k="Live sockets"
                    v={`${bin.available_cells.length} usable · ${bin.reserved_cells.length} tool`}
                  />
                )}
                <Row k="Clearance" v={`${bin.clearance_mm} mm`} />
              </tbody>
            </table>
          </div>

          <div>
            <div className="grp-label mb-3">Calibration</div>
            <table className="dtable">
              <tbody>
                <Row k="Corners" v={String(calibration.corners)} />
                <Row k="Reproj RMS" v={`${calibration.rms_px} px`} />
                <Row k="Tilt" v={calibration.tilt_deg != null ? `${calibration.tilt_deg}°` : "—"} />
                <Row k="Height" v={calibration.camera_height_mm ? `${calibration.camera_height_mm} mm` : "—"} />
              </tbody>
            </table>
          </div>

          <div>
            <div className="grp-label mb-3">Correction</div>
            <table className="dtable">
              <tbody>
                <Row
                  k="Method"
                  v={
                    reconstruction
                      ? manualPhysical
                        ? "Manual physical outline"
                        : "Two-view local silhouette"
                      : "Single-height parallax"
                  }
                />
                <Row k="Silhouette height" v={`${bin.silhouette_height_mm} mm`} />
                <Row k="Full tool height" v={bin.full_height_mm != null ? `${bin.full_height_mm} mm` : "not recorded"} />
                {reconstruction ? (
                  <>
                    <Row
                      k="Footprint"
                      v={`${reconstruction.reconstructed_major_extent_mm.toFixed(2)} × ${reconstruction.reconstructed_minor_extent_mm.toFixed(2)} mm`}
                    />
                    {manualPhysical ? (
                      <>
                        <Row
                          k="Manual change"
                          v={`${(reconstruction.manual_hausdorff_mm ?? 0).toFixed(2)} mm max`}
                        />
                        <Row
                          k="Source"
                          v={formatCorrectionMethod(reconstruction.source_method)}
                        />
                      </>
                    ) : (
                      <>
                        <Row
                          k="Local height"
                          v={`${(reconstruction.height_p05_mm ?? 0).toFixed(1)}–${(reconstruction.height_p95_mm ?? 0).toFixed(1)} mm`}
                        />
                        <Row
                          k="Boundary fit p95"
                          v={`${(reconstruction.boundary_p95_error_mm ?? 0).toFixed(2)} mm`}
                        />
                      </>
                    )}
                  </>
                ) : (
                  <Row k="Parallax" v={parallax ? `× ${parallax.toFixed(4)}` : "—"} />
                )}
              </tbody>
            </table>
            <div className="mt-4 flex flex-wrap gap-2">
              {downloads.map((kind) => (
                <a
                  key={kind}
                  className="btn btn-ghost text-xs px-4 py-2"
                  href={artifactUrl(files[kind])}
                  download
                >
                  {kind}
                </a>
              ))}
            </div>
          </div>
        </div>

        {hasLiveSession && (
          <section className="mt-8 border border-line bg-paper p-5">
            <div className="grp-label mb-2">Adjust bin</div>
            <p className="font-body text-sm text-muted mb-5 max-w-[72ch]">
              Regenerate from the accepted tool outline. Live grid keeps the complete stackable
              corral and adds functional sockets only where a full 42 mm cell fits outside the tool wall.
            </p>
            <div className="mb-4 grid grid-cols-3 gap-2">
              {(["pocket", "corral", "grid"] as BinStyle[]).map((style) => (
                <button
                  key={style}
                  type="button"
                  aria-pressed={binStyle === style}
                  className={`btn text-xs ${binStyle === style ? "border-teal text-teal" : "btn-ghost"}`}
                  disabled={regenerating || returning}
                  onClick={() => setBinStyle(style)}
                >
                  {style === "pocket" ? "Pocket" : style === "corral" ? "Stackable corral" : "Live grid"}
                </button>
              ))}
            </div>
            <div className="grid gap-4 md:grid-cols-3">
              <label className="block">
                <span className="grp-label block mb-2">
                  {binStyle === "pocket" ? "Pocket depth override (mm)" : "Tool recess depth override (mm)"}
                </span>
                <input
                  aria-label="Tool recess depth override"
                  className="mono-input w-full"
                  type="number"
                  min={0.1}
                  step={0.5}
                  placeholder="auto"
                  value={pocketDepth}
                  disabled={regenerating || returning}
                  onChange={(event) => setPocketDepth(event.target.value)}
                />
                <span className="font-mono text-[10px] text-muted block mt-2">
                  Current generated recess: {bin.pocket_depth_mm} mm
                </span>
              </label>
              <label className="block">
                <span className="grp-label block mb-2">Full tool height (mm)</span>
                <input
                  aria-label="Full tool height"
                  className="mono-input w-full"
                  type="number"
                  min={0.1}
                  step={0.5}
                  placeholder="optional"
                  value={fullToolHeight}
                  disabled={regenerating || returning}
                  onChange={(event) => setFullToolHeight(event.target.value)}
                />
                <span className="font-mono text-[10px] text-muted block mt-2">
                  Drives automatic recess depth; silhouette height stays {bin.silhouette_height_mm} mm.
                </span>
              </label>
              <label className="block">
                <span className="grp-label block mb-2">
                  Finished bin height override (mm)
                </span>
                <input
                  aria-label="Finished bin height override"
                  className="mono-input w-full"
                  type="number"
                  min={0.1}
                  step={1}
                  placeholder="auto"
                  value={finishedHeight}
                  disabled={regenerating || returning}
                  onChange={(event) => setFinishedHeight(event.target.value)}
                />
                <span className="font-mono text-[10px] text-muted block mt-2">
                  Current finished height: {bin.overall_height_mm} mm
                </span>
              </label>
            </div>
            <p className="font-mono text-[10px] text-muted mt-4">
              Finished height includes the stacking lip and snaps upward to a whole
              7 mm Gridfinity unit. Corral places its thin tool shelf at the selected
              recess depth and runs the separator to the stacking plane.
            </p>
            {controlError && (
              <p className="font-mono text-xs text-orange mt-4">
                Could not update bin: {controlError}
              </p>
            )}
            <div className="mt-5 flex flex-wrap gap-3">
              <button
                className="btn btn-primary"
                disabled={regenerating || returning || editingCutout}
                onClick={regenerate}
              >
                {regenerating ? "Regenerating…" : "Regenerate preview"}
              </button>
              {tool_poly && (
                <button
                  className="btn btn-ghost"
                  disabled={regenerating || returning || editingCutout}
                  onClick={() => setEditingCutout(true)}
                >
                  Edit physical cutout
                </button>
              )}
            </div>
          </section>
        )}

        {warnings.length > 0 && (
          <div className="mt-8">
            <div className="grp-label text-orange-text mb-2">Notes</div>
            <ul className="space-y-1">
              {warnings.map((w, i) => (
                <li key={i} className="font-mono text-xs text-muted">
                  — {w}
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="mt-10 flex items-center justify-between">
          <div className="flex flex-wrap gap-3">
            {hasLiveSession && (
              <button
                className="btn btn-ghost"
                disabled={returning || regenerating}
                onClick={backToEditor}
              >
                {returning ? "Returning…" : "Edit photo selection"}
              </button>
            )}
            <button className="btn" onClick={reset}>
              Trace another
            </button>
            {saved ? (
              <button className="btn btn-ghost text-teal" onClick={setLibrary}>
                ✓ In library →
              </button>
            ) : (
              <button
                className="btn btn-ghost"
                disabled={readiness.status === "block"}
                onClick={saveToLibrary}
              >
                Save to library
              </button>
            )}
          </div>
          <span className="font-mono text-xs text-muted" style={{ letterSpacing: "0.2em" }}>
            GRIDSHOT ◴
          </span>
        </div>
      </div>
    </div>
  );
}

function physicalEditorPolygon(tool: Poly): Poly {
  return {
    exterior: tool.exterior.map(([x, y]): [number, number] => [x, -y]),
    holes: tool.holes.map((ring) =>
      ring.map(([x, y]): [number, number] => [x, -y]),
    ),
  };
}

function formatCorrectionMethod(value?: string): string {
  if (value === "two_view_local_silhouette") return "Two-view local silhouette";
  if (value === "single_height_parallax") return "Single-height parallax";
  return value?.replaceAll("_", " ") ?? "Reconstructed outline";
}
function parseOptionalPositive(value: string, label: string): number | null {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const parsed = Number(trimmed);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive number or left blank for automatic.`);
  }
  return parsed;
}


function Row({ k, v }: { k: string; v: string }) {
  return (
    <tr>
      <td>{k}</td>
      <td>{v}</td>
    </tr>
  );
}
