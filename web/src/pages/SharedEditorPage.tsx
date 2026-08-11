import { useState } from "react";
import {
  sessionAddToLibrary,
  sessionClick,
  sessionEditHistory,
  sessionGenerate,
  type OutlineEditState,
  type OutlineVariant,
} from "../api";
import { OutlineCorrectionEditor } from "../components/OutlineCorrectionEditor";
import { ReadinessPanel } from "../components/ReadinessPanel";
import { useApp } from "../state";

/** Single-tool selection using the same correction surface as batch/library. */
export function Editor() {
  const { session, params, setTracing, setResult, setError, setLibrary, reset } = useApp();
  const [outlineCount, setOutlineCount] = useState(session?.polygon.exterior.length ?? 0);
  const [readiness, setReadiness] = useState(session?.readiness ?? null);
  const [score, setScore] = useState<number | null>(null);
  const [outlineVariant, setOutlineVariant] = useState<OutlineVariant>(
    session?.accepted_variant ?? "raw",
  );
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  if (!session || !params) return null;

  const activeSession = session;
  const activeParams = params;
  function changed(next: OutlineEditState) {
    setOutlineCount(next.polygon.exterior.length);
    if (next.score != null) setScore(next.score);
    setOutlineVariant(next.accepted_variant);
  }

  async function prompt(
    points: Parameters<typeof sessionClick>[1],
    box?: [number, number, number, number] | null,
  ) {
    const result = await sessionClick(activeSession.session, points, box);
    setReadiness(result.readiness);
    return result;
  }

  async function history(direction: "undo" | "redo") {
    const result = await sessionEditHistory(activeSession.session, direction);
    setReadiness(result.readiness);
    return result;
  }

  async function addLibrary() {
    setSaving(true);
    setActionError(null);
    try {
      await sessionAddToLibrary(
        activeSession.session, activeParams, outlineVariant,
      );
      setSaved(true);
    } catch (reason) {
      setActionError((reason as Error).message);
    } finally {
      setSaving(false);
    }
  }

  async function generate() {
    setTracing();
    try {
      setResult(
        await sessionGenerate(
          activeSession.session, activeParams, outlineVariant,
        ),
        activeParams,
      );
    } catch (reason) {
      setError((reason as Error).message);
    }
  }

  const blocked = outlineCount < 3 || saving || readiness?.status === "block";

  return (
    <div className="mx-auto max-w-container px-6 py-8">
      <div className="specline border-y border-line py-3 mb-6 flex flex-wrap gap-x-6 gap-y-2">
        <span>Correct and accept one tool</span>
        <span>{session.calibration.corners} corners · rms {session.calibration.rms_px}px</span>
        {score != null && <span>SAM {score.toFixed(3)}</span>}
      </div>

      <h1 className="titledev text-2xl mb-4">
        <span className="text-teal">SELECT</span>
        <span className="text-muted">•</span>
        <span>TOOL</span>
      </h1>
      <p className="font-body mb-6 max-w-[68ch]">
        Correct the photographed tool boundary here with an AI box or add/remove
        points. This changes segmentation before physical reconstruction. Vertex
        and opening edits belong to the physical cutout: generate first, or save
        to the library and edit the cutout there.
      </p>
      {actionError && <p className="font-mono text-xs text-orange mb-3">{actionError}</p>}
      {readiness && <ReadinessPanel readiness={readiness} className="mb-5" />}

      <OutlineCorrectionEditor
        session={session}
        onPrompt={prompt}
        onHistory={history}
        onChange={changed}
        onVariantChange={setOutlineVariant}
        title="Photo selection correction"
      />

      <div className="mt-5 flex flex-wrap items-center gap-3">
        <button className="btn" onClick={reset}>Back</button>
        <div className="flex-1" />
        {saved ? (
          <button className="btn btn-ghost text-teal" onClick={setLibrary}>✓ In library →</button>
        ) : (
          <button className="btn btn-ghost text-teal" disabled={blocked} onClick={addLibrary}>
            {saving ? "Saving…" : "Add to library"}
          </button>
        )}
        <button className="btn btn-primary" disabled={blocked} onClick={generate}>
          Generate bin
        </button>
      </div>
    </div>
  );
}
