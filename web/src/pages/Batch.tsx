import { useEffect, useRef, useState } from "react";
import { useApp } from "../state";
import {
  batchEditClick,
  batchEditHistory,
  batchEditSave,
  batchEditStart,
  cancelBatchJob,
  commitBatch,
  getBatchJob,
  listBatchJobs,
  postBatch,
  resumeBatchJob,
  reviewBatch,
  type BatchJob,
  type BatchPairSelection,
  type BatchReviewItem,
  type BatchReviewResult,
  type BatchSingleSelection,
  type OutlineEditSession,
  type OutlineVariant,
  type Poly,
} from "../api";
import { ReadinessPanel } from "../components/ReadinessPanel";
import { OutlineCorrectionEditor } from "../components/OutlineCorrectionEditor";
import { PhysicalCutoutEditor } from "../components/PhysicalCutoutEditor";

type ReviewPair = {
  a: number;
  b: number;
  iou?: number | null;
  score?: number;
  method?: string;
  gate?: string;
  reason?: string;
  confidence?: {
    level: "high" | "review" | "low";
    calibrated: boolean;
    score: number;
    inliers: number;
    inlier_ratio: number;
  };
  thickness_mm: number | null;
  manualThickness: string;
  source: "visual" | "manual" | "suggested";
};


const ACTIVE_BATCH_KEY = "gridshot.activeBatchSession";
function parsedThickness(value: string | undefined): number | null {
  if (!value?.trim()) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

/** Batch zip ingest: upload many tools shot two-photos each, review or repair
 *  the review candidates, then add only ready tools to the library. */
export function Batch() {
  const batch = useApp((s) => s.batch);
  const setBatch = useApp((s) => s.setBatch);
  const setLibrary = useApp((s) => s.setLibrary);
  const reset = useApp((s) => s.reset);
  const fileRef = useRef<HTMLInputElement>(null);
  const initialized = useRef<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [job, setJob] = useState<BatchJob | null>(null);
  const [recentJobs, setRecentJobs] = useState<BatchJob[]>([]);
  const [showOriginals, setShowOriginals] = useState(false);
  const [pairs, setPairs] = useState<ReviewPair[]>([]);
  const [unpaired, setUnpaired] = useState<number[]>([]);
  const [singleThickness, setSingleThickness] = useState<Record<number, string>>({});
  const [excluded, setExcluded] = useState<number[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [review, setReview] = useState<BatchReviewResult | null>(null);
  const [physicalOutlines, setPhysicalOutlines] = useState<Record<string, Poly>>({});
  const [physicalEditor, setPhysicalEditor] = useState<{
    key: string;
    label: string;
    polygon: Poly;
  } | null>(null);

  const [editor, setEditor] = useState<{ idx: number; sess: OutlineEditSession } | null>(null);
  const [editedThumbs, setEditedThumbs] = useState<Record<number, string>>({});

  useEffect(() => {
    let cancelled = false;
    listBatchJobs()
      .then((items) => {
        if (cancelled) return;
        setRecentJobs(items.filter((item) => !item.result?.committed));
        if (batch || job) return;
        const active = localStorage.getItem(ACTIVE_BATCH_KEY);
        const candidate = active
          ? items.find((item) => item.session === active)
          : undefined;
        if (!candidate) return;
        if (candidate.status === "ready" && candidate.result) {
          setBatch(candidate.result);
        } else {
          setJob(candidate);
        }
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!job || !["queued", "processing", "matching"].includes(job.status)) return;
    let cancelled = false;
    const poll = async () => {
      try {
        const next = await getBatchJob(job.session);
        if (cancelled) return;
        if (next.status === "ready" && next.result) {
          setBatch(next.result);
          setJob(null);
          return;
        }
        setJob(next);
      } catch (reason) {
        if (!cancelled) setErr((reason as Error).message);
      }
    };
    const timer = window.setInterval(poll, 750);
    void poll();
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [job?.session, job?.status, setBatch]);

  useEffect(() => {
    if (!batch || initialized.current === batch.session) return;
    initialized.current = batch.session;
    localStorage.setItem(ACTIVE_BATCH_KEY, batch.session);
    const draft = batch.draft?.selection;
    const accepted = (draft?.pairs ?? batch.pairs).map((pair) => {
      const suggested = batch.pairs.find((item) => (
        (item.a === pair.a && item.b === pair.b) ||
        (item.a === pair.b && item.b === pair.a)
      ));
      return {
        ...(suggested ?? pair),
        a: pair.a,
        b: pair.b,
        thickness_mm: pair.thickness_mm,
        manualThickness: pair.thickness_mm == null ? "" : String(pair.thickness_mm),
        source: suggested ? "visual" as const : "manual" as const,
      };
    });
    const committed = new Set(batch.committed_images ?? []);
    const used = new Set(accepted.flatMap((pair) => [pair.a, pair.b]));
    setPairs(accepted);
    setUnpaired(batch.images
      .map((image) => image.idx)
      .filter((idx) => !used.has(idx) && !committed.has(idx)));
    const singles = draft?.singles ?? [];
    setSingleThickness(Object.fromEntries(
      singles.map((single) => [single.idx, single.thickness_mm == null ? "" : String(single.thickness_mm)]),
    ));
    const included = new Set([...used, ...singles.map((single) => single.idx)]);
    setExcluded(draft
      ? batch.images
        .map((image) => image.idx)
        .filter((idx) => !included.has(idx) && !committed.has(idx))
      : []);
    setSelected(null);
    setReview(batch.draft?.review ?? null);
    setPhysicalOutlines(draft?.physical_outlines ?? {});
    setPhysicalEditor(null);
    setEditedThumbs({});
  }, [batch]);

  const image = (idx: number) => batch?.images.find((item) => item.idx === idx);
  const thumb = (idx: number) => {
    const value = image(idx);
    if (showOriginals) return value?.photo;
    return editedThumbs[idx] ?? value?.overlay ?? value?.thumb;
  };
  const name = (idx: number) => image(idx)?.name;
  const includedSingles = unpaired.filter((idx) => !excluded.includes(idx));
  const selectionCount = pairs.length + includedSingles.length;

  function invalidateReview() {
    setReview(null);
    setPhysicalOutlines({});
    setPhysicalEditor(null);
    setErr(null);
    setNotice(null);
  }

  async function openEditor(idx: number) {
    if (!batch) return;
    setBusy(true);
    setErr(null);
    try {
      setEditor({ idx, sess: await batchEditStart(batch.session, idx) });
    } catch (reason) {
      setErr((reason as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function saveEditor(outlineVariant: OutlineVariant) {
    if (!editor) return;
    try {
      const saved = await batchEditSave(editor.sess.session, outlineVariant);
      setEditedThumbs((current) => ({ ...current, [saved.idx]: saved.thumb }));
      setEditor(null);
      invalidateReview();
    } catch (reason) {
      setErr((reason as Error).message);
    }
  }

  function selections(): [BatchPairSelection[], BatchSingleSelection[]] {
    return [
      pairs.map((pair) => ({
        a: pair.a,
        b: pair.b,
        thickness_mm: parsedThickness(pair.manualThickness),
      })),
      includedSingles.map((idx) => ({
        idx,
        thickness_mm: parsedThickness(singleThickness[idx]),
      })),
    ];
  }

  async function upload(f: File) {
    setBusy(true);
    setErr(null);
    setBatch(null);
    initialized.current = null;
    setPairs([]);
    setUnpaired([]);
    setSingleThickness({});
    setExcluded([]);
    setSelected(null);
    setReview(null);
    try {
      const started = await postBatch(f);
      localStorage.setItem(ACTIVE_BATCH_KEY, started.session);
      setJob(started);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function cancelProcessing() {
    if (!job) return;
    setErr(null);
    try {
      setJob(await cancelBatchJob(job.session));
    } catch (reason) {
      setErr((reason as Error).message);
    }
  }

  async function resumeProcessing() {
    if (!job) return;
    setErr(null);
    try {
      setJob(await resumeBatchJob(job.session));
    } catch (reason) {
      setErr((reason as Error).message);
    }
  }

  function openJob(value: BatchJob) {
    localStorage.setItem(ACTIVE_BATCH_KEY, value.session);
    if (value.status === "ready" && value.result) {
      setBatch(value.result);
      setJob(null);
    } else {
      setJob(value);
    }
  }

  function startNewBatch() {
    localStorage.removeItem(ACTIVE_BATCH_KEY);
    initialized.current = null;
    setBatch(null);
    setJob(null);
    setErr(null);
    setNotice(null);
  }

  function pairImages(
    a: number,
    b: number,
    source: ReviewPair["source"] = "manual",
    iou?: number,
  ) {
    if (
      a === b ||
      !unpaired.includes(a) ||
      !unpaired.includes(b) ||
      excluded.includes(a) ||
      excluded.includes(b)
    ) return;
    setPairs((current) => [
      ...current,
      { a, b, iou, thickness_mm: null, manualThickness: "", source },
    ]);
    setUnpaired((current) => current.filter((idx) => idx !== a && idx !== b));
    setSelected(null);
    invalidateReview();
  }

  function unpair(pair: ReviewPair) {
    setPairs((current) => current.filter((item) => item !== pair));
    setUnpaired((current) => [...new Set([...current, pair.a, pair.b])].sort((a, b) => a - b));
    setSelected(null);
    invalidateReview();
  }

  function selectUnpaired(idx: number) {
    if (excluded.includes(idx)) return;
    if (selected == null) {
      setSelected(idx);
    } else if (selected === idx) {
      setSelected(null);
    } else {
      pairImages(selected, idx);
    }
  }

  function setPairThickness(pair: ReviewPair, value: string) {
    setPairs((current) => current.map((item) => (
      item === pair ? { ...item, manualThickness: value } : item
    )));
    invalidateReview();
  }

  function setOnePhotoThickness(idx: number, value: string) {
    setSingleThickness((current) => ({ ...current, [idx]: value }));
    invalidateReview();
  }

  function toggleExcluded(idx: number) {
    setExcluded((current) => (
      current.includes(idx)
        ? current.filter((item) => item !== idx)
        : [...current, idx]
    ));
    if (selected === idx) setSelected(null);
    invalidateReview();
  }

  function removeReviewTool(item: BatchReviewItem) {
    if (item.kind === "pair") {
      const pair = pairs.find((candidate) => (
        `pair:${candidate.a}:${candidate.b}` === item.key
      ));
      if (pair) {
        setPairs((current) => current.filter((candidate) => candidate !== pair));
        setUnpaired((current) => (
          [...new Set([...current, ...item.images])].sort((a, b) => a - b)
        ));
      }
    }
    setExcluded((current) => [...new Set([...current, ...item.images])]);
    setSelected(null);
    setReview(null);
    setPhysicalOutlines((current) => {
      const next = { ...current };
      delete next[item.key];
      return next;
    });
    setPhysicalEditor(null);
    setErr(null);
    setNotice(`Removed ${item.label} as one tool. Check the remaining tools before adding them.`);
  }

  async function checkReadiness(overrides: Record<string, Poly> = physicalOutlines) {
    if (!batch || selectionCount === 0) return;
    setBusy(true);
    setErr(null);
    try {
      const [pairSelections, singleSelections] = selections();
      setReview(await reviewBatch(
        batch.session,
        pairSelections,
        singleSelections,
        overrides,
      ));
    } catch (e) {
      setReview(null);
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function savePhysicalOutline(polygon: Poly) {
    if (!physicalEditor) return;
    const next = { ...physicalOutlines, [physicalEditor.key]: polygon };
    setPhysicalOutlines(next);
    setPhysicalEditor(null);
    await checkReadiness(next);
  }

  async function commit(readyOnly = false, discardBlocked = false) {
    if (
      !batch ||
      !review ||
      (!readyOnly && review.blocked > 0) ||
      (readyOnly && review.ready === 0)
    ) return;
    setBusy(true);
    setErr(null);
    setNotice(null);
    try {
      const [pairSelections, singleSelections] = selections();
      const outcome = await commitBatch(
        batch.session,
        pairSelections,
        singleSelections,
        physicalOutlines,
        readyOnly,
        discardBlocked,
      );
      if (outcome.partial) {
        const latest = await getBatchJob(batch.session);
        if (latest.result) {
          initialized.current = null;
          setBatch(latest.result);
        }
        setNotice(
          `Added ${outcome.added} ready tool${outcome.added === 1 ? "" : "s"} to the library. ` +
          `${outcome.remaining} blocked tool${outcome.remaining === 1 ? "" : "s"} remain in this draft.`,
        );
        return;
      }
      setBatch(null);
      localStorage.removeItem(ACTIVE_BATCH_KEY);
      setLibrary();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const pairItem = (pair: ReviewPair) => (
    review?.items.find((item) => item.key === `pair:${pair.a}:${pair.b}`)
  );
  const singleItem = (idx: number) => (
    review?.items.find((item) => item.key === `single:${idx}`)
  );

  const Readiness = ({ item }: { item?: BatchReviewItem }) => {
    if (!review) return <div className="text-[10px] text-muted">not checked</div>;
    if (!item) return null;
    return (
      <div className="space-y-1">
        <div className="font-mono text-[10px] text-muted">
          {item.thickness_mm != null
            ? `${item.thickness_mm} mm · ${item.thickness_source}`
            : item.status.replaceAll("_", " ")}
        </div>
        {item.reconstruction && (
          <div className="font-mono text-[10px] text-muted">
            local footprint {item.reconstruction.reconstructed_major_extent_mm.toFixed(2)} ×{" "}
            {item.reconstruction.reconstructed_minor_extent_mm.toFixed(2)} mm
          </div>
        )}
        {item.warnings.length > 0 && (
          <details className="font-mono text-[10px] text-orange-text max-w-64">
            <summary>{item.warnings.length} capture warning{item.warnings.length === 1 ? "" : "s"}</summary>
            {item.warnings.map((warning, index) => (
              <div key={index}>— {warning}</div>
            ))}
          </details>
        )}
        <ReadinessPanel readiness={item.readiness} compact />
        {item.physical_outline && (
          <button
            type="button"
            className="font-mono text-[10px] text-teal hover:underline"
            onClick={() => setPhysicalEditor({
              key: item.key,
              label: item.label,
              polygon: item.physical_outline!,
            })}
          >
            {physicalOutlines[item.key] ? "✓ edit physical cutout" : "edit physical cutout"}
          </button>
        )}
        {item.reason && (
          <div className="font-mono text-[10px] text-orange-text max-w-64">
            {item.reason}
          </div>
        )}
        {item.status !== "ready" && (
          <button
            type="button"
            className="font-mono text-[10px] text-orange hover:underline"
            onClick={() => removeReviewTool(item)}
          >
            remove this tool from batch
          </button>
        )}
      </div>
    );
  };

  const Thumb = ({ idx, selectable = false, primary = false }: {
    idx: number;
    selectable?: boolean;
    primary?: boolean;
  }) => {
    const warnings = image(idx)?.warnings ?? [];
    return (
      <div className="text-center w-24">
        <button
          type="button"
          className={`relative border-2 ${selectable && selected === idx ? "border-teal" : "border-transparent"}`}
          style={{ borderRadius: 2 }}
          disabled={!selectable}
          onClick={() => selectable && selectUnpaired(idx)}
        >
          <img src={thumb(idx)} alt={name(idx)} className="w-24 h-24 object-contain bg-field border border-line" style={{ borderRadius: 2 }} />
          {primary && <span className="absolute left-1 top-1 bg-black/80 px-1 font-mono text-[8px] text-teal">PRIMARY</span>}
        </button>
        <div className="font-mono text-[9px] text-muted truncate w-24" title={name(idx)}>{name(idx)}</div>
        {warnings.length > 0 && <div className="font-mono text-[8px] text-orange-text">{warnings.length} warning{warnings.length === 1 ? "" : "s"}</div>}
        <button type="button" className="font-mono text-[9px] text-teal hover:underline" onClick={() => openEditor(idx)}>
          correct outline
        </button>
      </div>
    );
  };

  const suggestions = batch?.flagged.filter(
    (candidate) => (
      unpaired.includes(candidate.a) &&
      unpaired.includes(candidate.b) &&
      !excluded.includes(candidate.a) &&
      !excluded.includes(candidate.b)
    ),
  ) ?? [];

  return (
    <div className="mx-auto max-w-container px-6 py-12">
      <header className="mb-8 flex items-end justify-between">
        <h1 className="titledev text-3xl">
          <span className="text-teal">BATCH</span> <span className="text-muted">ZIP</span>
        </h1>
        <button className="btn" onClick={reset}>← Capture</button>
      </header>

      {err && <div className="panel mb-6 border-orange"><p className="font-mono text-sm">{err}</p></div>}
      {notice && <div className="panel mb-6 border-teal"><p className="font-mono text-sm">{notice}</p></div>}

      {!batch && job && (
        <div className="panel space-y-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <div className="grp-label">Batch {job.status}</div>
              <div className="font-mono text-xs text-muted mt-1">
                {job.processed_images} / {job.total_images} images · {job.succeeded_images} captured
                {job.failed_images ? ` · ${job.failed_images} failed` : ""}
              </div>
              {job.current_name && <div className="font-mono text-xs text-teal mt-1">Processing {job.current_name}</div>}
              {job.error && <div className="font-mono text-xs text-orange-text mt-1">{job.error}</div>}
            </div>
            <div className="flex gap-2">
              {["queued", "processing", "matching"].includes(job.status) && (
                <button className="btn" disabled={job.cancel_requested} onClick={cancelProcessing}>
                  {job.cancel_requested ? "Stopping…" : "Cancel"}
                </button>
              )}
              {job.can_resume && <button className="btn btn-primary" onClick={resumeProcessing}>Resume</button>}
              <button className="btn" onClick={startNewBatch}>Start new</button>
            </div>
          </div>
          <div className="h-2 overflow-hidden bg-field border border-line" style={{ borderRadius: 2 }}>
            <div className="h-full bg-teal" style={{ width: `${Math.round(job.processed_images / Math.max(1, job.total_images) * 100)}%` }} />
          </div>
          <div className="grid gap-1 sm:grid-cols-2 lg:grid-cols-3">
            {job.entries.map((entry) => (
              <div key={entry.name} className="font-mono text-[10px] text-muted truncate" title={entry.reason ?? entry.name}>
                <span className={entry.status === "failed" ? "text-orange-text" : entry.status === "complete" ? "text-teal" : ""}>[{entry.status}]</span> {entry.name}
              </div>
            ))}
          </div>
        </div>
      )}

      {!batch && !job && (
        <div className="panel space-y-4">
          <div className="grp-label mb-3">Upload a zip of tool photos</div>
          <p className="font-body mb-4 max-w-[60ch]">
            Shoot each tool <strong>two photos</strong> with the camera moved and the tool
            and mat fixed. A one-photo tool can be saved only after you enter its measured
            thickness. GridShot checks every included tool before anything reaches the library.
          </p>
          <p className="font-mono text-[10px] text-muted">Up to 200 images · 1 GiB ZIP · 4 GiB expanded · duplicate filenames are rejected</p>
          <button className="btn btn-primary" disabled={busy} onClick={() => fileRef.current?.click()}>
            {busy ? "Uploading…" : "Choose zip"}
          </button>
          <input ref={fileRef} type="file" accept=".zip,application/zip" className="hidden"
            onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])} />
        </div>
      )}

      {!batch && !job && recentJobs.length > 0 && (
        <div className="panel mt-6">
          <div className="grp-label mb-3">Recent batch drafts</div>
          <div className="space-y-2">
            {recentJobs.slice(0, 5).map((value) => (
              <button key={value.session} type="button" className="w-full border border-line p-3 text-left hover:border-teal" onClick={() => openJob(value)}>
                <div className="flex justify-between gap-3 font-mono text-xs">
                  <span>{value.result?.draft ? "review draft" : value.status}</span>
                  <span className="text-muted">{value.processed_images}/{value.total_images} images</span>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {batch && (
        <div className="space-y-6">
          <div className="specline font-mono text-xs text-muted">
            {pairs.length} paired tools · {includedSingles.length} one-photo tools
            {excluded.length ? ` · ${excluded.length} excluded` : ""}
            {batch.failed.length ? ` · ${batch.failed.length} failed` : ""}
          </div>

          <div className="flex flex-wrap gap-2">
            <button className={`btn ${showOriginals ? "" : "btn-primary"}`} onClick={() => setShowOriginals(false)}>
              Mask overlays
            </button>
            <button className={`btn ${showOriginals ? "btn-primary" : ""}`} onClick={() => setShowOriginals(true)}>
              Original photos
            </button>
            <button className="btn ml-auto" onClick={startNewBatch}>Start new batch</button>
          </div>

          {batch.matcher.warning && (
            <div className="panel border-orange">
              <div className="font-mono text-xs text-orange-text">{batch.matcher.warning}</div>
            </div>
          )}

          {pairs.length > 0 && (
            <div className="panel">
              <div className="grp-label mb-1">Paired tools — verify thickness before save</div>
              <p className="font-mono text-xs text-muted mb-3">
                Automatic thickness is calculated during readiness review. Enter a measured
                override only when the automatic solve is unavailable or untrustworthy.
              </p>
              <div className="flex flex-wrap gap-6">
                {pairs.map((pair) => {
                  const item = pairItem(pair);
                  return (
                    <div key={`${pair.a}-${pair.b}`} className="flex gap-2 items-start">
                      <Thumb idx={pair.a} primary={item ? item.primary_image === pair.a : true} />
                      <Thumb idx={pair.b} primary={item?.primary_image === pair.b} />
                      <div className="font-mono text-xs min-w-36 space-y-1">
                        <Readiness item={item} />
                        <div className="text-[9px] text-muted">
                          {pair.source === "visual"
                            ? pair.method + " · score " + pair.score?.toFixed(1)
                            : pair.source}
                        </div>
                        {pair.confidence && (
                          <div className="text-[9px] text-teal max-w-44">
                            {pair.confidence.level} prefill confidence · strict gate, not calibrated probability
                          </div>
                        )}
                        {pair.reason && (
                          <div className="text-[9px] text-muted max-w-44">{pair.reason}</div>
                        )}
                        <label className="block text-[9px] text-muted">
                          measured override (mm)
                          <input
                            className="field mt-1 w-28"
                            type="number"
                            min="0.1"
                            step="0.1"
                            value={pair.manualThickness}
                            onChange={(event) => setPairThickness(pair, event.target.value)}
                            placeholder="automatic"
                          />
                        </label>
                        <button className="text-orange hover:underline" onClick={() => unpair(pair)}>
                          split pair
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {suggestions.length > 0 && (
            <div className="panel">
              <div className="grp-label mb-1 text-amber-500">Pair suggestions</div>
              <p className="font-mono text-xs text-muted mb-3">
                Shape overlap is only a review hint; it is never accepted as tool identity automatically.
              </p>
              <div className="flex flex-wrap gap-6">
                {suggestions.map((candidate) => (
                  <div key={`${candidate.a}-${candidate.b}`} className="flex gap-2 items-center">
                    <Thumb idx={candidate.a} /><Thumb idx={candidate.b} />
                    <div className="font-mono text-xs">
                      <div className="text-muted">shape hint {candidate.iou}</div>
                      {candidate.reason && <div className="text-[9px] text-muted max-w-36">{candidate.reason}</div>}
                      <button
                        className="px-2 py-0.5 border border-line text-teal"
                        style={{ borderRadius: 2 }}
                        onClick={() => pairImages(candidate.a, candidate.b, "suggested", candidate.iou)}
                      >
                        same tool
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {unpaired.length > 0 && (
            <div className="panel">
              <div className="grp-label mb-1">Unpaired photos</div>
              <p className="font-mono text-xs text-muted mb-3">
                {selected == null
                  ? "Select the cleaner outline first, then its matching view. For a true one-photo tool, enter its measured thickness. Explicitly exclude photos you do not want saved."
                  : `Selected ${name(selected)} — choose its matching view, or tap it again to cancel.`}
              </p>
              <div className="flex flex-wrap gap-4">
                {unpaired.map((idx) => {
                  const isExcluded = excluded.includes(idx);
                  return (
                    <div key={idx} className={`border p-2 ${isExcluded ? "border-line opacity-60" : "border-transparent"}`}>
                      <div className="flex gap-2 items-start">
                        <Thumb idx={idx} selectable={!isExcluded} primary={singleItem(idx)?.primary_image === idx} />
                        <div className="font-mono text-xs space-y-1">
                          {isExcluded
                            ? <div className="text-[10px] text-muted">excluded</div>
                            : <Readiness item={singleItem(idx)} />}
                          <label className="block text-[9px] text-muted">
                            measured thickness (mm)
                            <input
                              className="field mt-1 w-28"
                              type="number"
                              min="0.1"
                              step="0.1"
                              disabled={isExcluded}
                              value={singleThickness[idx] ?? ""}
                              onChange={(event) => setOnePhotoThickness(idx, event.target.value)}
                              placeholder="required"
                            />
                          </label>
                          <button
                            className={isExcluded ? "text-teal hover:underline" : "text-orange hover:underline"}
                            onClick={() => toggleExcluded(idx)}
                          >
                            {isExcluded ? "include" : "exclude"}
                          </button>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {batch.failed.length > 0 && (
            <div className="panel">
              <div className="grp-label mb-2 text-orange-text">Not segmented</div>
              <ul className="font-mono text-xs text-muted">{batch.failed.map((f, i) => <li key={i}>— {f.name}: {f.reason}</li>)}</ul>
            </div>
          )}

          {review && (
            <div className={`panel ${review.blocked ? "border-orange" : "border-teal"}`}>
              <div className="font-mono text-sm">
                {review.ready} ready · {review.blocked} blocked
              </div>
              <div className="font-mono text-xs text-muted mt-1">
                {review.blocked
                  ? "Add the ready tools now and keep the blocked tools as a draft, remove a blocked tool as one unit, or remove every blocked tool while adding the ready set."
                  : "All included tools passed readiness. Commit will add them as one batch."}
              </div>
            </div>
          )}

          <div className="flex flex-wrap gap-3">
            <button
              className="btn"
              disabled={busy || selectionCount === 0}
              onClick={() => checkReadiness()}
            >
              {busy ? "Checking…" : `Check ${selectionCount} tools`}
            </button>
            {review && review.blocked > 0 && review.ready > 0 ? (
              <>
                <button
                  className="btn btn-primary"
                  disabled={busy}
                  onClick={() => commit(true, false)}
                >
                  {busy
                    ? "Adding…"
                    : `Add ${review.ready} ready · keep ${review.blocked} blocked draft →`}
                </button>
                <button
                  className="btn"
                  disabled={busy}
                  onClick={() => commit(true, true)}
                >
                  Remove {review.blocked} blocked & add {review.ready} ready
                </button>
              </>
            ) : (
              <button
                className="btn btn-primary"
                disabled={
                  busy ||
                  !review ||
                  review.blocked > 0 ||
                  review.ready !== selectionCount
                }
                onClick={() => commit()}
              >
                {busy ? "Adding…" : `Add ${selectionCount} ready tools to library →`}
              </button>
            )}
          </div>
        </div>
      )}
      {editor && (
        <div
          className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto p-4"
          style={{ background: "rgba(0,0,0,0.7)" }}
          onClick={() => setEditor(null)}
        >
          <div onClick={(event) => event.stopPropagation()}>
            <OutlineCorrectionEditor
              session={editor.sess}
              onPrompt={(points, box) => batchEditClick(editor.sess.session, points, box)}
              onHistory={(direction) => batchEditHistory(editor.sess.session, direction)}
              onSave={(_state, outlineVariant) => saveEditor(outlineVariant)}
              onCancel={() => setEditor(null)}
              title={`Batch correction · ${name(editor.idx)}`}
              saveLabel="Save to batch"
            />
          </div>
        </div>
      )}
      {physicalEditor && (
        <div
          className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto p-4"
          style={{ background: "rgba(0,0,0,0.7)" }}
          onClick={() => setPhysicalEditor(null)}
        >
          <div className="w-full max-w-[900px]" onClick={(event) => event.stopPropagation()}>
            <div className="font-mono text-xs text-knockout mb-2 truncate">
              Batch cutout · {physicalEditor.label}
            </div>
            <PhysicalCutoutEditor
              initial={physicalEditor.polygon}
              busy={busy}
              onSave={savePhysicalOutline}
              onCancel={() => setPhysicalEditor(null)}
            />
          </div>
        </div>
      )}
    </div>
  );
}
