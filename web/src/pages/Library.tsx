import { useEffect, useMemo, useRef, useState } from "react";
import {
  composeLibrary,
  createLibraryBackup,
  deleteLibraryTool,
  downloadLibraryArchive,
  drawerPreviewGlb,
  exportDrawer,
  getLibraryOutline,
  getLibraryPhotoOutline,
  libraryEditClick,
  libraryEditHistory,
  libraryEditSave,
  libraryEditStart,
  listLibrary,
  updateLibraryTool,
  type ComposeResult,
  type LibraryEditResult,
  type LibraryTool,
  type OutlineVariant,
  type PhotoOutline,
  type Poly,
} from "../api";
import { OutlineCorrectionEditor } from "../components/OutlineCorrectionEditor";
import { PhysicalCutoutEditor } from "../components/PhysicalCutoutEditor";
import { CombineEditor } from "../components/CombineEditor";
import { DrawerViewer } from "../components/DrawerViewer";
import { PhotoLightbox } from "../components/PhotoLightbox";
import { ReadinessBadge } from "../components/ReadinessPanel";

const PAL = ["#d65a54", "#5ab478", "#548cd6", "#e6be46", "#c85ac8", "#50c8c8", "#e69646", "#a050d6"];

/** Tool library: individually-captured tools composed into one drawer. Select
 *  tools saved from separate captures, pick a drawer size, and nest them —
 *  building a big set from small, accurate, fully-on-mat captures. */
export function Library() {
  const [tools, setTools] = useState<LibraryTool[]>([]);
  const [sel, setSel] = useState<Set<string>>(new Set());
  const [cols, setCols] = useState(8);
  const [rows, setRows] = useState(6);
  const [overallHeight, setOverallHeight] = useState<number | "">("");
  const [composed, setComposed] = useState<ComposeResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [editing, setEditing] = useState<{
    id: string;
    label: string;
    polygon: Poly;
  } | null>(null);
  const [sam, setSam] = useState<{ id: string; sess: LibraryEditResult } | null>(null);
  const [combining, setCombining] = useState(false);
  const [viewing, setViewing] = useState<{ t: LibraryTool; data: PhotoOutline } | null>(null);
  const [drawerPreviewUrl, setDrawerPreviewUrl] = useState<string | null>(null);
  const [drawerPreviewBusy, setDrawerPreviewBusy] = useState(false);
  const [drawerPreviewError, setDrawerPreviewError] = useState<string | null>(null);
  const [libraryNotice, setLibraryNotice] = useState<string | null>(null);
  const drawerPreviewUrlRef = useRef<string | null>(null);
  const drawerPreviewSequence = useRef(0);

  function clearDrawerPreview() {
    drawerPreviewSequence.current += 1;
    if (drawerPreviewUrlRef.current) {
      URL.revokeObjectURL(drawerPreviewUrlRef.current);
      drawerPreviewUrlRef.current = null;
    }
    setDrawerPreviewUrl(null);
    setDrawerPreviewError(null);
    setDrawerPreviewBusy(false);
  }

  function invalidateComposition() {
    clearDrawerPreview();
    setComposed(null);
  }

  async function openView(t: LibraryTool) {
    try {
      const data = await getLibraryPhotoOutline(t.id);
      if (data.has_photo) setViewing({ t, data });
      else toggle(t.id); // no photo → clicking just selects (old behaviour)
    } catch {
      toggle(t.id);
    }
  }
  // thumbnails live at a static URL; bump a version per tool so the <img>
  // reloads the regenerated image after an outline edit (else it's cached)
  const [bust, setBust] = useState<Record<string, number>>({});
  // prefer the real-photo crop when present (tells similar outlines apart)
  const thumbSrc = (t: LibraryTool) => {
    const base = t.photo_thumb ?? t.thumb;
    return bust[t.id] ? `${base}?v=${bust[t.id]}` : base;
  };

  async function openOutline(id: string) {
    try {
      const polygon = await getLibraryOutline(id);
      if (!polygon) throw new Error("this tool has no physical cutout");
      setEditing({
        id,
        label: tools.find((tool) => tool.id === id)?.label || id,
        polygon,
      });
    } catch (e) {
      alert((e as Error).message);
    }
  }
  async function saveOutline(polygon: Poly) {
    if (!editing) return;
    setBusy(true);
    try {
      await patch(editing.id, {
        outline: polygon,
        edit_source: "physical",
      });
      setEditing(null);
      invalidateComposition();
    } catch (e) {
      alert("Save failed: " + (e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function openSam(id: string) {
    try {
      setSam({ id, sess: await libraryEditStart(id) });
    } catch (e) {
      alert((e as Error).message);
    }
  }
  async function saveSam(outlineVariant: OutlineVariant) {
    if (!sam) return;
    try {
      const saved = await libraryEditSave(sam.sess.session, outlineVariant);
      setTools((current) =>
        current.map((tool) => (tool.id === saved.id ? { ...tool, ...saved } : tool)),
      );
      setBust((value) => ({ ...value, [sam.id]: Date.now() }));
      setSam(null);
      invalidateComposition();
    } catch (e) {
      alert("Save failed: " + (e as Error).message);
    }
  }

  const refresh = () => listLibrary().then(setTools).catch(() => setTools([]));
  useEffect(() => { refresh(); }, []);
  useEffect(() => () => {
    if (drawerPreviewUrlRef.current) URL.revokeObjectURL(drawerPreviewUrlRef.current);
  }, []);

  const toggle = (id: string) => {
    if (tools.find((tool) => tool.id === id)?.readiness.status === "block") return;
    invalidateComposition();
    setSel((s) => {
      const n = new Set(s);
      n.has(id) ? n.delete(id) : n.add(id);
      return n;
    });
  };

  async function remove(id: string) {
    await deleteLibraryTool(id);
    setSel((s) => { const n = new Set(s); n.delete(id); return n; });
    invalidateComposition();
    refresh();
  }

  async function patch(id: string, changes: Parameters<typeof updateLibraryTool>[1]) {
    const u = await updateLibraryTool(id, changes);
    setTools((ts) => ts.map((x) => (x.id === id ? { ...x, ...u } : x)));
    if (u.readiness.status === "block") {
      setSel((current) => {
        const next = new Set(current);
        next.delete(id);
        return next;
      });
    }
    if ("outline" in changes || "thickness_mm" in changes || "silhouette_height_mm" in changes)
      setBust((b) => ({ ...b, [id]: Date.now() })); // outline/thumb regenerated
    invalidateComposition();
  }

  async function compose() {
    if (!sel.size) return;
    setBusy(true);
    try {
      clearDrawerPreview();
      setComposed(null);
      setComposed(await composeLibrary(
        [...sel],
        cols,
        rows,
        overallHeight === "" ? null : overallHeight,
      ));
    } finally {
      setBusy(false);
    }
  }

  async function generateDrawerPreview() {
    if (!composed) return;
    const sequence = ++drawerPreviewSequence.current;
    setDrawerPreviewBusy(true);
    setDrawerPreviewError(null);
    try {
      const blob = await drawerPreviewGlb(
        [...sel],
        cols,
        rows,
        overallHeight === "" ? null : overallHeight,
      );
      if (sequence !== drawerPreviewSequence.current) return;
      const nextUrl = URL.createObjectURL(blob);
      if (drawerPreviewUrlRef.current) URL.revokeObjectURL(drawerPreviewUrlRef.current);
      drawerPreviewUrlRef.current = nextUrl;
      setDrawerPreviewUrl(nextUrl);
    } catch (reason) {
      if (sequence === drawerPreviewSequence.current) {
        setDrawerPreviewError((reason as Error).message);
      }
    } finally {
      if (sequence === drawerPreviewSequence.current) setDrawerPreviewBusy(false);
    }
  }

  async function exportLibrary() {
    setLibraryNotice(null);
    try {
      await downloadLibraryArchive();
      setLibraryNotice("Portable library archive downloaded.");
    } catch (reason) {
      setLibraryNotice((reason as Error).message);
    }
  }

  async function backupLibrary() {
    setLibraryNotice(null);
    try {
      const saved = await createLibraryBackup();
      setLibraryNotice(
        `Saved ${saved.filename} with ${saved.tool_count} tool${saved.tool_count === 1 ? "" : "s"}.`,
      );
    } catch (reason) {
      setLibraryNotice((reason as Error).message);
    }
  }

  const idColor = (id: string) => PAL[[...sel].indexOf(id) % PAL.length] || "#888";
  const drawerColors = useMemo(
    () => Object.fromEntries(
      (composed?.layout.placed ?? []).map((placement) => [
        placement.bin_id,
        idColor(placement.bin_id),
      ]),
    ),
    [composed, sel],
  );

  return (
    <div className="mx-auto max-w-container px-4 py-8 sm:px-6 sm:py-12">
      <header className="mb-8 flex flex-wrap items-end justify-between gap-4">
        <div>
          <div className="grp-label mb-2">{tools.length} saved</div>
          <h1 className="titledev text-3xl">
            <span className="text-teal">TOOL</span> <span className="text-muted">LIBRARY</span>
          </h1>
        </div>
        <div className="flex flex-wrap items-center justify-end gap-2">
          <button className="btn btn-ghost" onClick={backupLibrary}>
            Create backup
          </button>
          <button className="btn btn-ghost" onClick={exportLibrary}>
            Export library
          </button>
        </div>
      </header>

      {libraryNotice && (
        <div className="mb-5 border border-line bg-paper-2 px-4 py-3 font-mono text-xs text-knockout">
          {libraryNotice}
        </div>
      )}

      {tools.length === 0 && (
        <div className="panel">
          <p className="font-body">
            No saved tools yet. Trace a tool (or a small batch), then use{" "}
            <strong>Save to library</strong> on the result. Saved tools compose into a drawer here.
          </p>
        </div>
      )}

      {tools.length > 0 && (
        <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_320px]">
          {/* tool grid */}
          <div className="panel min-w-0 p-4 sm:p-6">
            <div className="grp-label mb-4">01 · Select tools</div>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-[repeat(auto-fill,minmax(220px,1fr))]">
              {tools.map((t) => {
                const on = sel.has(t.id);
                const blocked = t.readiness.status === "block";
                return (
                  <div
                    key={t.id}
                    className="border bg-paper-2 overflow-hidden"
                    style={{ borderRadius: 2, borderColor: on ? "var(--c-teal)" : "var(--c-line)" }}
                  >
                    <div className="relative bg-field">
                      <img
                        src={thumbSrc(t)}
                        alt={t.label || t.id}
                        className="w-full aspect-square object-contain p-3"
                        style={{ cursor: "zoom-in" }}
                        title="View the source photo and accepted selection"
                        onClick={() => openView(t)}
                      />
                      <button
                        className="absolute top-2 left-2 w-7 h-7 flex items-center justify-center font-mono text-sm border bg-field"
                        style={{
                          borderRadius: 2, lineHeight: 1,
                          borderColor: on ? "var(--c-teal)" : "var(--c-line)",
                          color: on ? "#50c8c8" : "var(--c-knockout)",
                        }}
                        disabled={blocked}
                        onClick={(e) => { e.stopPropagation(); toggle(t.id); }}
                        title={
                          blocked
                            ? t.readiness.checks
                              .filter((check) => check.status === "block")
                              .map((check) => check.message)
                              .join("\n")
                            : "Select for compose or combine"
                        }
                      >
                        {on ? "✓" : ""}
                      </button>
                      <button
                        className="absolute top-2 right-2 w-7 h-7 bg-field border border-line text-knockout hover:text-gold font-mono text-base"
                        onClick={(e) => { e.stopPropagation(); remove(t.id); }}
                        title="Delete tool"
                      >
                        ×
                      </button>
                    </div>
                    <div className="p-3 space-y-3 text-field" onClick={(e) => e.stopPropagation()}>
                      <label className="block">
                        <span className="font-mono text-[10px] uppercase text-muted">Tool name</span>
                        <input
                          className="w-full bg-transparent border-b border-line font-mono text-sm text-field py-1 outline-none placeholder:text-muted"
                          defaultValue={t.label}
                          placeholder="Unnamed tool"
                          onBlur={(e) => e.target.value !== t.label && patch(t.id, { label: e.target.value })}
                        />
                      </label>
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-mono text-[10px] uppercase text-muted">Print readiness</span>
                        <ReadinessBadge readiness={t.readiness} />
                      </div>
                      <div className="grid grid-cols-3 gap-2">
                        {(["pocket", "corral", "grid"] as const).map((style) => (
                          <button
                            key={style}
                            type="button"
                            aria-pressed={t.bin_style === style}
                            className={`border px-2 py-1 font-mono text-[10px] ${t.bin_style === style ? "border-teal text-teal" : "border-line text-muted"}`}
                            style={{ borderRadius: 2 }}
                            onClick={() => patch(t.id, { bin_style: style }).catch(() => {})}
                          >
                            {style === "pocket" ? "Pocket" : style === "corral" ? "Corral" : "Live grid"}
                          </button>
                        ))}
                      </div>
                      <div className="flex flex-wrap items-center justify-between gap-2 font-mono text-[10px] text-muted">
                        <span>
                          {t.grid_x}×{t.grid_y}u · {t.derived_overall_height_mm ?? "—"}mm tall
                          {t.bin_style === "grid" ? ` · ${t.derived_available_cells.length} live` : ""}
                        </span>
                        <button
                          className={`px-2 py-1 border ${t.finger_hole ? "text-teal border-teal" : "text-muted border-line"}`}
                          style={{ borderRadius: 2 }}
                          title={t.bin_style !== "pocket"
                            ? "enclosed access lobe inside the separator for easier tool removal"
                            : "finger scallop to lift the recessed tool out (full-depth pockets need it)"}
                          onClick={() => patch(t.id, { finger_hole: !t.finger_hole }).catch(() => {})}
                        >
                          Finger access: {t.finger_hole ? "on" : "off"}
                        </button>
                      </div>
                      <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
                        <label className="min-w-0">
                          <span className="block font-mono text-[9px] uppercase text-muted">Silhouette height</span>
                          <input
                            aria-label="Tool thickness in millimetres"
                            className="mono-input min-w-0 !px-2 !py-1 !text-sm"
                            type="number" step={0.5} min={0}
                            defaultValue={t.thickness_mm}
                            onBlur={(e) => Number(e.target.value) !== t.thickness_mm && patch(t.id, { thickness_mm: Number(e.target.value) })}
                          />
                        </label>
                        <label className="min-w-0">
                          <span className="block font-mono text-[9px] uppercase text-muted">Full height</span>
                          <input
                            aria-label="Full tool height in millimetres"
                            className="mono-input min-w-0 !px-2 !py-1 !text-sm"
                            type="number" step={0.5} min={0.1}
                            placeholder="optional"
                            defaultValue={t.full_height_mm ?? ""}
                            onBlur={(e) => {
                              const value = e.target.value === "" ? null : Number(e.target.value);
                              if (value !== t.full_height_mm) patch(t.id, { full_height_mm: value }).catch(() => {});
                            }}
                          />
                        </label>
                        <label className="min-w-0">
                          <span className="block font-mono text-[9px] uppercase text-muted">
                            {t.bin_style === "pocket" ? "Depth" : "Recess"}
                          </span>
                          <input
                            key={`d-${t.id}-${t.derived_key}`}
                            aria-label="Tool recess depth in millimetres"
                            className="mono-input min-w-0 !px-2 !py-1 !text-sm"
                            style={{ color: t.pocket_depth_mm == null ? "var(--c-muted)" : undefined }}
                            type="number" step={0.5} min={0.1}
                            defaultValue={t.pocket_depth_mm ?? t.derived_pocket_depth_mm ?? ""}
                            title={t.pocket_depth_mm == null ? "Automatic recess depth; type to override" : "Manual override; clear for automatic"}
                            onBlur={(e) => {
                              const v = e.target.value === "" ? null : Number(e.target.value);
                              const matchesDerived =
                                v != null &&
                                t.derived_pocket_depth_mm != null &&
                                v === t.derived_pocket_depth_mm;
                              patch(t.id, { pocket_depth_mm: matchesDerived ? null : v }).catch(() => {});
                            }}
                          />
                        </label>
                        <label className="min-w-0">
                          <span className="block font-mono text-[9px] uppercase text-muted">Clearance</span>
                          <input
                            aria-label="Pocket clearance in millimetres"
                            className="mono-input min-w-0 !px-2 !py-1 !text-sm"
                            type="number" step={0.25} min={0}
                            defaultValue={t.clearance_mm}
                            onBlur={(e) => Number(e.target.value) !== t.clearance_mm && patch(t.id, { clearance_mm: Number(e.target.value) })}
                          />
                        </label>
                      </div>
                      <div className="flex flex-wrap gap-x-4 gap-y-1 border-t border-line pt-2">
                        <button className="font-mono text-[10px] text-teal hover:underline" onClick={() => openOutline(t.id)}>
                          ✎ Edit physical cutout · R{t.outline_revision}
                        </button>
                        {t.has_photo && (
                          <button className="font-mono text-[10px] text-teal hover:underline" onClick={() => openSam(t.id)}>
                            ◐ Correct photo selection
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* compose controls */}
          <div className="panel h-fit min-w-0 p-4 sm:p-6">
            <div className="grp-label mb-4">02 · Compose drawer</div>
            <div className="space-y-4">
              <label className="block">
                <span className="font-mono text-xs text-muted">Width (cells)</span>
                <input className="mono-input w-full" type="number" min={1} value={cols}
                  onChange={(e) => {
                    const next = Math.max(1, Math.round(Number(e.target.value)));
                    if (next !== cols) invalidateComposition();
                    setCols(next);
                  }} />
              </label>
              <label className="block">
                <span className="font-mono text-xs text-muted">Depth (cells)</span>
                <input className="mono-input w-full" type="number" min={1} value={rows}
                  onChange={(e) => {
                    const next = Math.max(1, Math.round(Number(e.target.value)));
                    if (next !== rows) invalidateComposition();
                    setRows(next);
                  }} />
              </label>
              <label className="block">
                <span className="font-mono text-xs text-muted">Overall height (mm)</span>
                <input className="mono-input w-full" type="number" min={0} step={1} value={overallHeight}
                  placeholder="auto (per bin)"
                  onChange={(e) => {
                    clearDrawerPreview();
                    setOverallHeight(e.target.value === "" ? "" : Number(e.target.value));
                  }} />
                <span className="font-mono text-[10px] text-muted">blank = each bin its own height; set for level tops</span>
              </label>
              <button className="btn btn-primary w-full" disabled={!sel.size || busy} onClick={compose}>
                {busy ? "…" : `Compose ${sel.size} tool${sel.size === 1 ? "" : "s"} (separate bins)`}
              </button>
              {sel.size >= 2 && (
                <button className="btn w-full" onClick={() => setCombining(true)}>
                  ⧉ Combine into ONE bin…
                </button>
              )}
            </div>

            {composed && (
              <div className="mt-6">
                <div className="grp-label mb-2">
                  Layout · uses {composed.layout.used_cols}×{composed.layout.used_rows}u
                </div>
                <div className="border border-line bg-field p-2" style={{ borderRadius: 2 }}>
                  <svg viewBox={`-0.2 -0.2 ${cols + 0.4} ${rows + 0.4}`} className="w-full">
                    {Array.from({ length: cols + 1 }, (_, c) => (
                      <line key={"c" + c} x1={c} y1={0} x2={c} y2={rows} stroke="#3a4046" strokeWidth={0.03} />
                    ))}
                    {Array.from({ length: rows + 1 }, (_, r) => (
                      <line key={"r" + r} x1={0} y1={r} x2={cols} y2={r} stroke="#3a4046" strokeWidth={0.03} />
                    ))}
                    {composed.layout.placed.map((p) => (
                      <rect key={p.bin_id} x={p.col + 0.04} y={p.row + 0.04}
                        width={p.grid_x - 0.08} height={p.grid_y - 0.08}
                        fill={idColor(p.bin_id) + "55"} stroke={idColor(p.bin_id)} strokeWidth={0.05} rx={0.08} />
                    ))}
                  </svg>
                </div>
                {composed.layout.overflow.length > 0 && (
                  <p className="font-mono text-xs text-muted mt-2">
                    {composed.layout.overflow.length} didn't fit — enlarge the drawer.
                  </p>
                )}
                <button
                  className="btn w-full mt-4"
                  disabled={drawerPreviewBusy || composed.layout.placed.length === 0}
                  onClick={generateDrawerPreview}
                >
                  {drawerPreviewBusy
                    ? "Generating 3D…"
                    : drawerPreviewUrl
                      ? "Regenerate 3D preview"
                      : "Generate 3D preview"}
                </button>
                {drawerPreviewError && (
                  <p className="mt-2 font-mono text-xs text-orange" role="alert">
                    {drawerPreviewError}
                  </p>
                )}
                {drawerPreviewUrl && (
                  <div className="mt-3">
                    <div
                      className="aspect-square w-full overflow-hidden border border-line bg-field"
                      style={{ borderRadius: 2 }}
                    >
                      <DrawerViewer url={drawerPreviewUrl} binColors={drawerColors} />
                    </div>
                    <p className="mt-2 font-mono text-[10px] text-muted">
                      Exact bins seated in the full {cols}×{rows} Gridfinity socket grid · drag to orbit · scroll to zoom
                    </p>
                    <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1">
                      {composed.layout.placed.map((placement) => {
                        const tool = composed.tools.find((item) => item.id === placement.bin_id);
                        return (
                          <span key={placement.bin_id} className="inline-flex items-center gap-1 font-mono text-[10px] text-muted">
                            <span className="h-2 w-2" style={{ background: drawerColors[placement.bin_id] }} />
                            {tool?.label || placement.bin_id}
                          </span>
                        );
                      })}
                    </div>
                  </div>
                )}
                <button
                  className="btn btn-primary w-full mt-4"
                  onClick={() => exportDrawer([...sel], cols, rows, overallHeight === "" ? null : overallHeight).catch(() => {})}
                >
                  ↓ Export drawer (3MF + layout)
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {editing && (
        <div
          className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto p-4"
          style={{ background: "rgba(0,0,0,0.6)" }}
          onClick={() => setEditing(null)}
        >
          <div className="w-full max-w-[900px]" onClick={(e) => e.stopPropagation()}>
            <div className="font-mono text-xs text-knockout mb-2 truncate">
              Library cutout · {editing.label}
            </div>
            <PhysicalCutoutEditor
              initial={editing.polygon}
              busy={busy}
              onSave={saveOutline}
              onCancel={() => setEditing(null)}
            />
          </div>
        </div>
      )}

      {sam && (
        <div
          className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto p-4"
          style={{ background: "rgba(0,0,0,0.6)" }}
          onClick={() => setSam(null)}
        >
          <div onClick={(e) => e.stopPropagation()}>
            <OutlineCorrectionEditor
              session={sam.sess}
              onPrompt={(points, box) => libraryEditClick(sam.sess.session, points, box)}
              onHistory={(direction) => libraryEditHistory(sam.sess.session, direction)}
              onSave={(_state, outlineVariant) => saveSam(outlineVariant)}
              onCancel={() => setSam(null)}
              title="Library photo selection correction"
            />
          </div>
        </div>
      )}

      {viewing && (
        <div
          className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto p-4"
          style={{ background: "rgba(0,0,0,0.7)" }}
          onClick={() => setViewing(null)}
        >
          <div onClick={(e) => e.stopPropagation()}>
            <PhotoLightbox
              data={viewing.data}
              label={viewing.t.label}
              onClose={() => setViewing(null)}
              onCutout={() => { const id = viewing.t.id; setViewing(null); openOutline(id); }}
              onRefine={() => { const id = viewing.t.id; setViewing(null); openSam(id); }}
            />
          </div>
        </div>
      )}

      {combining && (
        <div
          className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto p-4"
          style={{ background: "rgba(0,0,0,0.6)" }}
          onClick={() => setCombining(false)}
        >
          <div className="w-full max-w-[980px]" onClick={(e) => e.stopPropagation()}>
            <CombineEditor
              ids={[...sel]}
              overallHeight={overallHeight === "" ? null : overallHeight}
              lip={true}
              onClose={() => setCombining(false)}
            />
          </div>
        </div>
      )}
    </div>
  );
}
