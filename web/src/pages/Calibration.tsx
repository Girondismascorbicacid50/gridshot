import { useEffect, useMemo, useState } from "react";
import {
  calibrateIntrinsics,
  deleteAllDeviceProfiles,
  deleteDeviceProfile,
  getDeviceProfiles,
  getMats,
  type DeviceProfileSummary,
  type IntrinsicsCalibrationResult,
  type Mat,
} from "../api";
import { useApp } from "../state";

const MIN_VIEWS = 8;
const RECOMMENDED_VIEWS = 12;

export function Calibration() {
  const reset = useApp((state) => state.reset);
  const [mats, setMats] = useState<Mat[]>([]);
  const [profiles, setProfiles] = useState<DeviceProfileSummary[]>([]);
  const [matId, setMatId] = useState("");
  const [name, setName] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [result, setResult] = useState<IntrinsicsCalibrationResult | null>(
    null,
  );

  useEffect(() => {
    Promise.all([getMats(), getDeviceProfiles()])
      .then(([matValues, profileValues]) => {
        const verified = matValues.filter((mat) => mat.verified);
        setMats(verified);
        setProfiles(profileValues);
        setMatId((current) => current || verified[0]?.mat_id || "");
      })
      .catch((reason: Error) => setError(reason.message));
  }, []);

  const captureSummary = useMemo(() => {
    if (!result) return null;
    const signature = result.capture_signature;
    const camera = [signature.device_make, signature.device_model]
      .filter(Boolean)
      .join(" ");
    return {
      camera: camera || "Camera metadata unavailable",
      lens: signature.lens_model || "Lens metadata unavailable",
      resolution: `${signature.image_size[0]} × ${signature.image_size[1]}`,
      orientation: `${signature.orientation_deg}°`,
      zoom:
        signature.digital_zoom_ratio == null
          ? "Metadata unavailable"
          : `${signature.digital_zoom_ratio.toFixed(2)}×`,
    };
  }, [result]);

  async function runCalibration() {
    if (!matId || files.length < MIN_VIEWS) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const value = await calibrateIntrinsics(files, matId, name);
      setResult(value);
      setProfiles((current) => [
        value.profile,
        ...current.filter(
          (profile) => profile.device_id !== value.profile.device_id,
        ),
      ]);
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function removeProfile(profile: DeviceProfileSummary) {
    const confirmed = window.confirm(
      `Delete camera profile "${profile.device_id}"? Existing saved traces ` +
        "will keep their recorded calibration, but new captures will stop " +
        "using this profile.",
    );
    if (!confirmed) return;

    setDeleting(profile.device_id);
    setError(null);
    try {
      await deleteDeviceProfile(profile.device_id);
      setProfiles((current) =>
        current.filter((item) => item.device_id !== profile.device_id),
      );
      if (result?.profile.device_id === profile.device_id) {
        setResult(null);
      }
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setDeleting(null);
    }
  }

  async function removeAllProfiles() {
    const confirmed = window.confirm(
      `Delete all ${profiles.length} camera profiles? Existing saved traces ` +
        "will keep their recorded calibration, but new captures will use " +
        "estimated intrinsics until you recalibrate.",
    );
    if (!confirmed) return;

    setDeleting("all");
    setError(null);
    try {
      await deleteAllDeviceProfiles();
      setProfiles([]);
      setResult(null);
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setDeleting(null);
    }
  }

  return (
    <div className="mx-auto max-w-container px-6 py-12">
      <header className="mb-10">
        <div className="specline border-y border-line py-3 mb-8 flex items-center gap-6">
          <span>Accuracy setup · Camera intrinsics</span>
          <button className="ml-auto text-teal hover:underline" onClick={reset}>
            ← Back to trace
          </button>
        </div>
        <div className="grp-label mb-2">One-time capture setup</div>
        <h1 className="titledev text-3xl leading-none">
          <span className="text-teal">CALIBRATE</span>
          <span>CAMERA</span>
        </h1>
        <p className="font-body text-lg max-w-[66ch] mt-4">
          Measure the lens distortion for one exact camera, lens, orientation,
          resolution, and zoom. GridShot will select this profile automatically
          and will abstain when a later photo does not match.
        </p>
      </header>

      {error && (
        <div className="panel mb-8 border-orange" role="alert">
          <div className="grp-label text-orange-text mb-2">
            Calibration failed
          </div>
          <p className="font-mono text-sm">{error}</p>
        </div>
      )}

      <div className="grid gap-8 lg:grid-cols-[0.9fr_1.1fr]">
        <section className="panel">
          <div className="grp-label mb-4">01 · Photograph the mat</div>
          <ol className="space-y-5">
            <Guide n="A" title="Lock one capture setup">
              Use the same rear camera, lens, orientation, resolution, and zoom
              for every photo. Do not crop, edit, or mix camera modes.
            </Guide>
            <Guide n="B" title="Shoot 12–20 varied views">
              Move around the flat verified mat. Include centered, edge, near,
              far, and gently tilted views so the board covers the whole frame.
            </Guide>
            <Guide n="C" title="Keep every view usable">
              Keep the board sharp and fully visible, avoid glare, and do not
              change zoom after the first shot. More varied views beat repeated
              photos from one position.
            </Guide>
          </ol>
          <div className="mt-6 border-t border-line pt-5 font-mono text-xs text-muted">
            Profiles are immutable. Recalibrating the same capture setup creates
            a new revision; prior traces retain the revision they used.
          </div>
        </section>

        <section className="panel">
          <div className="grp-label mb-4">02 · Build profile</div>
          <div className="space-y-5">
            <label className="block">
              <span className="font-mono text-xs block mb-1">
                Verified calibration mat
              </span>
              <select
                className="mono-input"
                value={matId}
                onChange={(event) => setMatId(event.target.value)}
                disabled={busy}
              >
                {mats.length === 0 && (
                  <option value="">No verified mat available</option>
                )}
                {mats.map((mat) => (
                  <option key={mat.mat_id} value={mat.mat_id}>
                    {mat.mat_id} · {mat.paper.toUpperCase()}
                  </option>
                ))}
              </select>
            </label>

            <label className="block">
              <span className="font-mono text-xs block mb-1">
                Profile label <span className="text-muted">(optional)</span>
              </span>
              <input
                className="mono-input"
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="e.g. shop phone main camera"
                disabled={busy}
              />
            </label>

            <label className="block">
              <span className="font-mono text-xs block mb-1">
                Calibration photos
              </span>
              <input
                className="mono-input file:font-mono file:text-xs"
                type="file"
                accept="image/*,.heic,.heif"
                multiple
                disabled={busy}
                onChange={(event) =>
                  setFiles(Array.from(event.target.files ?? []))
                }
              />
              <span className="font-mono text-xs text-muted block mt-2">
                {files.length} selected · {MIN_VIEWS} minimum ·{" "}
                {RECOMMENDED_VIEWS}+ recommended
              </span>
            </label>

            {files.length > 0 && (
              <div className="border border-line bg-paper-2 px-3 py-2 max-h-36 overflow-auto">
                {files.map((file, index) => (
                  <div
                    key={`${file.name}-${index}`}
                    className="font-mono text-xs py-1 flex gap-3"
                  >
                    <span className="text-muted w-6">
                      {String(index + 1).padStart(2, "0")}
                    </span>
                    <span className="truncate">{file.name}</span>
                  </div>
                ))}
              </div>
            )}

            <button
              className="btn btn-primary"
              disabled={
                busy || !matId || files.length < MIN_VIEWS
              }
              onClick={runCalibration}
            >
              {busy ? "Calibrating…" : "Create camera profile"}
            </button>
          </div>
        </section>
      </div>

      {result && captureSummary && (
        <section className="panel mt-8 border-teal" aria-live="polite">
          <div className="flex flex-wrap items-start gap-4 mb-6">
            <div>
              <div className="grp-label mb-1">Profile ready</div>
              <h2 className="font-display font-bold text-xl">
                {result.profile.device_id}
              </h2>
            </div>
            <span className="badge text-teal border-teal ml-auto">
              Revision {result.profile.revision}
            </span>
          </div>
          <div className="grid gap-8 md:grid-cols-2">
            <table className="dtable">
              <tbody>
                <tr>
                  <td>Camera</td>
                  <td>{captureSummary.camera}</td>
                </tr>
                <tr>
                  <td>Lens</td>
                  <td>{captureSummary.lens}</td>
                </tr>
                <tr>
                  <td>Resolution</td>
                  <td>{captureSummary.resolution}</td>
                </tr>
                <tr>
                  <td>Orientation</td>
                  <td>{captureSummary.orientation}</td>
                </tr>
                <tr>
                  <td>Digital zoom</td>
                  <td>{captureSummary.zoom}</td>
                </tr>
              </tbody>
            </table>
            <table className="dtable">
              <tbody>
                <tr>
                  <td>Views used</td>
                  <td>
                    {result.views_used} / {result.views_uploaded}
                  </td>
                </tr>
                <tr>
                  <td>Reprojection RMS</td>
                  <td>{result.profile.reproj_rms_px.toFixed(3)} px</td>
                </tr>
                <tr>
                  <td>Mat</td>
                  <td>{result.profile.mat_id}</td>
                </tr>
              </tbody>
            </table>
          </div>
          {result.warnings.length > 0 && (
            <ul className="mt-5 font-mono text-xs text-orange-text list-disc pl-5">
              {result.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          )}
          <button className="btn btn-ghost mt-6" onClick={reset}>
            Trace a tool
          </button>
        </section>
      )}

      <section className="mt-10">
        <div className="mb-3 flex items-center gap-4">
          <div className="grp-label">Saved camera profiles</div>
          {profiles.length > 0 && (
            <button
              className="ml-auto font-mono text-xs uppercase text-orange-text hover:underline disabled:opacity-40"
              disabled={deleting !== null}
              onClick={removeAllProfiles}
            >
              {deleting === "all" ? "Deleting…" : "Delete all"}
            </button>
          )}
        </div>
        {profiles.length === 0 ? (
          <p className="font-mono text-xs text-muted">
            No calibrated capture setups yet.
          </p>
        ) : (
          <div className="grid gap-3 md:grid-cols-2">
            {profiles.map((profile) => (
              <div
                key={profile.device_id}
                className="border border-line p-4 flex items-start gap-4"
              >
                <div>
                  <div className="font-mono text-sm">{profile.device_id}</div>
                  <div className="specline mt-1">
                    {profile.image_size[0]}×{profile.image_size[1]} ·{" "}
                    {profile.orientation_deg}° · RMS{" "}
                    {profile.reproj_rms_px.toFixed(3)} px
                  </div>
                </div>
                <div className="ml-auto flex flex-col items-end gap-2">
                  <span className="badge">R{profile.revision}</span>
                  <button
                    className="font-mono text-xs uppercase text-orange-text hover:underline disabled:opacity-40"
                    disabled={deleting !== null}
                    onClick={() => removeProfile(profile)}
                  >
                    {deleting === profile.device_id ? "Deleting…" : "Delete"}
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

function Guide({
  n,
  title,
  children,
}: {
  n: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <li className="grid grid-cols-[2rem_1fr] gap-3">
      <span className="font-mono text-xs text-teal border border-teal w-8 h-8 flex items-center justify-center">
        {n}
      </span>
      <div>
        <div className="font-mono text-sm mb-1">{title}</div>
        <p className="font-body text-sm text-muted">{children}</p>
      </div>
    </li>
  );
}
