import { useEffect } from "react";
import { forgetActiveSession, loadActiveSession, useApp } from "./state";
import { Upload } from "./pages/Upload";
import { Editor } from "./pages/SharedEditorPage";
import { Result } from "./pages/Result";
import { Library } from "./pages/Library";
import { Batch } from "./pages/Batch";
import { Calibration } from "./pages/Calibration";
import { AppNavigation } from "./components/AppNavigation";
import { getSession, type Session, type TraceResult } from "./api";

const DEFAULT_PARAMS = {
  thickness: 4,
  full_height: null,
  clearance: 1.0,
  bin_style: "pocket" as const,
  depth: null,
  overall_height: null,
  finger_hole: true,
  lip: true,
};

export function App() {
  const view = useApp((s) => s.view);
  const setResult = useApp((s) => s.setResult);
  const setEditor = useApp((s) => s.setEditor);

  // deep-links: ?project=<id> reopens a stored result; ?session=<id> the editor
  useEffect(() => {
    const q = new URLSearchParams(location.search);
    const proj = q.get("project");
    const stored = loadActiveSession();
    const sess = q.get("session") ?? stored?.session;
    if (proj) {
      fetch(`/api/result/${proj}`)
        .then((r) => (r.ok ? r.json() : Promise.reject()))
        .then((r: TraceResult) => setResult(r))
        .catch(() => {});
    } else if (sess) {
      getSession(sess)
        .then((s: Session) => setEditor(
          s,
          stored?.session === sess ? stored.params : DEFAULT_PARAMS,
        ))
        .catch(() => forgetActiveSession());
    }
  }, [setResult, setEditor]);

  let page;
  if (view === "result") page = <Result />;
  else if (view === "library") page = <Library />;
  else if (view === "batch") page = <Batch />;
  else if (view === "calibration") page = <Calibration />;
  else if (view === "editor") page = <Editor />;
  else if (view === "tracing") page = <Tracing />;
  else page = <Upload />;

  return (
    <>
      <AppNavigation />
      {page}
    </>
  );
}

function Tracing() {
  return (
    <div className="mx-auto max-w-container px-6 py-24 text-center">
      <div className="grp-label mb-4">Working</div>
      <h1 className="titledev text-2xl justify-center">
        <span className="text-teal">SEGMENT</span>
        <span className="text-muted">•</span>
        <span>CALIBRATE</span>
        <span className="text-muted">•</span>
        <span>GENERATE</span>
      </h1>
      <p className="font-mono text-xs text-muted mt-6" style={{ letterSpacing: "0.18em" }}>
        RECOVERING 1:1 OUTLINE · CORRECTING PARALLAX · BUILDING BIN
      </p>
    </div>
  );
}
