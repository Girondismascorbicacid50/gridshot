// Typed client for the GridShot backend. Hand-written — ~a handful of endpoints.

export type Ring = [number, number][];
export interface Poly {
  exterior: Ring;
  holes: Ring[];
}

export type OutlineVariant = "raw" | "cleaned";

export interface OutlineCleanup {
  available: boolean;
  recommended: OutlineVariant;
  noise_mm: number;
  radius_mm: number;
  straightened: boolean;
  max_shift_cap_mm: number;
  symdiff_mm2: number;
  mean_shift_mm: number;
  max_shift_mm: number;
  area_ratio: number;
  reason?: string;
}

export interface OutlineEditDiagnostics {
  iou_with_previous: number;
  area_change_pct: number;
  iou_with_initial: number;
  area_change_from_initial_pct: number;
  mask_area_px: number;
  vertex_count: number;
  hole_count: number;
}

export interface OutlineEditState {
  outline: Ring;
  polygon: Poly;
  cleaned_polygon: Poly;
  smooth_polygon: Poly; // compatibility alias
  cleanup: OutlineCleanup;
  accepted_variant: OutlineVariant;
  points: ClickPoint[];
  score?: number;
  revision: number;
  operation: string;
  diagnostics: OutlineEditDiagnostics;
  iou_with_previous: number;
  area_change_pct: number;
  can_undo: boolean;
  can_redo: boolean;
  history_index: number;
  history_length: number;
}

export interface OutlineEditSession extends OutlineEditState {
  session: string;
  display: string;
  width: number;
  height: number;
  raw?: Poly;
  corrected?: Poly;
}

export type ReadinessStatus = "pass" | "review" | "block";
export type ReadinessSource =
  | "calibration"
  | "segmentation"
  | "cleanup"
  | "outline"
  | "thickness"
  | "printer"
  | "generation"
  | "provenance";

export interface ReadinessCheck {
  code: string;
  status: ReadinessStatus;
  source: ReadinessSource;
  message: string;
  confidence?: number | null;
}

export interface ReadinessReport {
  status: ReadinessStatus;
  checks: ReadinessCheck[];
  metrics: Record<string, number>;
}

export interface ArtifactProvenance {
  flow: "single" | "batch" | "legacy";
  mat_id: string | null;
  device_profile_id: string | null;
  device_profile_revision: number | null;
  intrinsics_source: string | null;
  capture_signature: CaptureSignature | null;
  thickness_source: "manual" | "automatic" | "legacy" | "unknown";
  source_images: string[];
  warnings: string[];
}

export interface FootprintReconstruction {
  method: "two_view_local_silhouette" | string;
  scalar_height_mm?: number;
  scalar_residual_mm2?: number;
  boundary_mean_error_mm?: number;
  boundary_p95_error_mm?: number;
  height_p05_mm?: number;
  height_median_mm?: number;
  height_p95_mm?: number;
  scalar_major_extent_mm?: number;
  reconstructed_major_extent_mm: number;
  reconstructed_minor_extent_mm: number;
  area_change_from_scalar_pct?: number;
  source_method?: string;
  manual_iou_with_previous?: number;
  manual_area_change_pct?: number;
  manual_hausdorff_mm?: number;
  physical_area_mm2?: number;
}

export type BinStyle = "pocket" | "corral" | "grid";

export interface TraceResult {
  project: string;
  bin: {
    grid: [number, number];
    height_u: number;
    overall_height_mm: number;
    bin_style: BinStyle;
    pocket_depth_mm: number;
    pocket_depth_override_mm: number | null;
    overall_height_override_mm: number | null;
    thickness_mm: number;
    silhouette_height_mm: number;
    full_height_mm: number | null;
    clearance_mm: number;
    lip: boolean;
    derivation_key: string;
    reserved_cells: [number, number][];
    available_cells: [number, number][];
  };
  calibration: {
    corners: number;
    rms_px: number;
    tilt_deg: number | null;
    camera_height_mm: number | null;
    nadir_xy_mm: [number, number] | null;
    mat_id: string;
    device_profile_id: string | null;
    device_profile_revision: number | null;
    intrinsics_source: string | null;
    capture_signature: CaptureSignature | null;
  };
  calibration_model: unknown;
  tool_poly: Poly | null;
  pocket_poly: Poly | null;
  raw_poly: Poly | null;
  corrected_poly: Poly | null;
  reconstruction: FootprintReconstruction | null;
  physical_outline_edit?: FootprintReconstruction | null;
  warnings: string[];
  readiness: ReadinessReport;
  provenance: ArtifactProvenance;
  files: Record<string, string>;
}

export interface Health {
  status: string;
  segserver: boolean;
  mats: string[];
}

export interface Mat {
  mat_id: string;
  paper: string;
  verified: boolean;
  scale_x: number;
  scale_y: number;
}

export async function getHealth(): Promise<Health> {
  const r = await fetch("/api/health");
  if (!r.ok) throw new Error("health check failed");
  return r.json();
}

export async function getMats(): Promise<Mat[]> {
  const r = await fetch("/api/mats");
  if (!r.ok) throw new Error("mat list failed");
  return r.json();
}

export interface CaptureSignature {
  schema_version: "capture.v1";
  device_make: string | null;
  device_model: string | null;
  lens_model: string | null;
  image_size: [number, number];
  orientation_deg: 0 | 90 | 180 | 270;
  mirrored: boolean;
  focal_mm: number | null;
  focal_35mm: number | null;
  digital_zoom_ratio: number | null;
}

export interface DeviceProfileSummary {
  device_id: string;
  revision: number;
  created_at: string;
  device_make: string | null;
  device_model: string | null;
  lens_model: string | null;
  image_size: [number, number];
  orientation_deg: 0 | 90 | 180 | 270;
  focal_mm: number | null;
  focal_35mm: number | null;
  digital_zoom_ratio: number | null;
  mat_id: string | null;
  n_views: number | null;
  reproj_rms_px: number;
}

export interface IntrinsicsCalibrationResult {
  profile: DeviceProfileSummary;
  capture_signature: CaptureSignature;
  views_uploaded: number;
  views_used: number;
  warnings: string[];
}

export async function getDeviceProfiles(): Promise<DeviceProfileSummary[]> {
  const r = await fetch("/api/device-profiles", { cache: "no-store" });
  if (!r.ok) throw new Error("camera profile list failed");
  return r.json();
}

async function deleteProfileRequest(path: string): Promise<unknown> {
  const r = await fetch(path, { method: "DELETE" });
  const data = await r.json().catch(() => ({ detail: r.statusText }));
  if (!r.ok) {
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : "camera profile deletion failed",
    );
  }
  return data;
}

export async function deleteDeviceProfile(deviceId: string): Promise<void> {
  await deleteProfileRequest(
    `/api/device-profiles/${encodeURIComponent(deviceId)}`,
  );
}

export async function deleteAllDeviceProfiles(): Promise<number> {
  const data = (await deleteProfileRequest("/api/device-profiles")) as {
    deleted: number;
  };
  return data.deleted;
}

export async function calibrateIntrinsics(
  files: File[],
  matId: string,
  name?: string,
): Promise<IntrinsicsCalibrationResult> {
  const fd = new FormData();
  files.forEach((file) => fd.append("files", file));
  fd.append("mat_id", matId);
  if (name?.trim()) fd.append("name", name.trim());
  const r = await fetch("/api/calibration/intrinsics", {
    method: "POST",
    body: fd,
  });
  const data = await r.json().catch(() => ({ detail: r.statusText }));
  if (!r.ok) {
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : "camera calibration failed",
    );
  }
  return data as IntrinsicsCalibrationResult;
}

export interface ClickPoint {
  x: number;
  y: number;
  label: number; // 1 fg, 0 bg
}

export interface Session extends OutlineEditState {
  session: string;
  display: string;
  width: number;
  height: number;
  calibration: {
    corners: number;
    rms_px: number;
    tilt_deg: number | null;
    device_profile_id: string | null;
    device_profile_revision: number | null;
    intrinsics_source: string | null;
  };
  has_photo2: boolean;
  warnings: string[];
  readiness: ReadinessReport;
}

export interface DrawerPlacement {
  bin_id: string;
  col: number;
  row: number;
  grid_x: number;
  grid_y: number;
  rotated: boolean;
}

export interface BatchImage {
  idx: number;
  name: string;
  thumb: string;
  overlay: string;
  photo: string;
  outline_thumb: string;
  warnings: string[];
  readiness: ReadinessReport | null;
}

export interface BatchDraft {
  version: number;
  updated_ts: number;
  selection: {
    pairs: BatchPairSelection[];
    singles: BatchSingleSelection[];
    physical_outlines: Record<string, Poly>;
  };
  review: BatchReviewResult;
}

export interface BatchResult {
  session: string;
  artifacts: string;
  images: BatchImage[];
  pairs: {
    a: number;
    b: number;
    iou: number | null;
    score: number;
    thickness_mm: number | null;
    method: string;
    gate: string;
    reason?: string;
    confidence?: {
      level: "high" | "review" | "low";
      calibrated: boolean;
      score: number;
      inliers: number;
      inlier_ratio: number;
    };
  }[];
  flagged: { a: number; b: number; iou: number; reason?: string }[];
  singles: number[];
  matcher: {
    method: string | null;
    gate: Record<string, number | string> | null;
    warning: string | null;
  };
  failed: { name: string; reason: string }[];
  draft: BatchDraft | null;
  committed?: boolean;
  committed_images?: number[];
  partial_commits?: number;
  commit_result?: BatchCommitResult;
}

export type BatchJobStatus =
  | "queued"
  | "processing"
  | "matching"
  | "paused"
  | "cancelled"
  | "failed"
  | "ready";

export interface BatchJob {
  session: string;
  status: BatchJobStatus;
  phase: string;
  total_images: number;
  processed_images: number;
  succeeded_images: number;
  failed_images: number;
  current_name: string | null;
  error: string | null;
  can_resume: boolean;
  cancel_requested: boolean;
  entries: {
    name: string;
    status: "pending" | "processing" | "complete" | "failed";
    reason: string | null;
  }[];
  result: BatchResult | null;
  created_ts: number;
  updated_ts: number;
}

async function batchJobRequest(url: string, method = "GET"): Promise<BatchJob> {
  const r = await fetch(url, { method, cache: "no-store" });
  const data = await r.json().catch(() => ({ detail: r.statusText }));
  if (!r.ok) throw new Error(data.detail ?? "batch job request failed");
  return data as BatchJob;
}

export async function postBatch(zip: File): Promise<BatchJob> {
  const fd = new FormData();
  fd.append("file", zip);
  const r = await fetch("/api/batch", { method: "POST", body: fd });
  const data = await r.json().catch(() => ({ detail: r.statusText }));
  if (!r.ok) {
    throw new Error(data.detail ?? "batch upload failed");
  }
  return data as BatchJob;
}

export const getBatchJob = (session: string) =>
  batchJobRequest(`/api/batch/${session}`);

export const cancelBatchJob = (session: string) =>
  batchJobRequest(`/api/batch/${session}/cancel`, "POST");

export const resumeBatchJob = (session: string) =>
  batchJobRequest(`/api/batch/${session}/resume`, "POST");

export async function listBatchJobs(): Promise<BatchJob[]> {
  const r = await fetch("/api/batch-jobs", { cache: "no-store" });
  if (!r.ok) throw new Error("batch jobs could not be loaded");
  return (await r.json()).jobs as BatchJob[];
}

export async function batchEditStart(
  batchSession: string,
  idx: number,
): Promise<OutlineEditSession> {
  const r = await fetch(
    `/api/batch/${batchSession}/image/${idx}/edit`,
    { method: "POST" },
  );
  if (!r.ok) {
    const d = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(d.detail ?? "batch editor failed to open");
  }
  return r.json();
}

export function batchEditClick(
  session: string,
  points: ClickPoint[],
  box?: [number, number, number, number] | null,
): Promise<OutlineEditState> {
  return editorPrompt(`/api/batch/edit/${session}/click`, points, box);
}

export function batchEditOutline(
  session: string,
  polygon: Poly,
): Promise<OutlineEditState> {
  return editorJson(`/api/batch/edit/${session}/outline`, polygon);
}

export function batchEditHistory(
  session: string,
  direction: "undo" | "redo",
): Promise<OutlineEditState> {
  return editorJson(`/api/batch/edit/${session}/history/${direction}`);
}

export interface BatchEditSaveResult {
  idx: number;
  revision: number;
  thumb: string;
  readiness: ReadinessReport;
}

export async function batchEditSave(
  session: string,
  outlineVariant: OutlineVariant,
): Promise<BatchEditSaveResult> {
  const body = new FormData();
  body.append("outline_variant", outlineVariant);
  const r = await fetch(`/api/batch/edit/${session}/save`, {
    method: "POST",
    body,
  });
  if (!r.ok) {
    const d = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(d.detail ?? "batch outline save failed");
  }
  return r.json();
}

export interface BatchPairSelection {
  a: number;
  b: number;
  thickness_mm: number | null;
}

export interface BatchSingleSelection {
  idx: number;
  thickness_mm: number | null;
}

export interface BatchReviewItem {
  key: string;
  kind: "pair" | "single";
  images: number[];
  label: string;
  primary_image: number;
  status: "ready" | "needs_pair" | "needs_outline" | "needs_thickness" | "failed";
  thickness_mm: number | null;
  thickness_source: "automatic" | "manual" | null;
  reason: string | null;
  warnings: string[];
  readiness: ReadinessReport;
  reconstruction: FootprintReconstruction | null;
  physical_outline?: Poly | null;
  library_id?: string;
}

export interface BatchReviewResult {
  items: BatchReviewItem[];
  ready: number;
  blocked: number;
}

export interface BatchCommitResult {
  added: number;
  committed: boolean;
  partial: boolean;
  remaining: number;
  discarded: number;
  items: BatchReviewItem[];
}

function batchError(detail: unknown, fallback: string): Error {
  if (typeof detail === "string") return new Error(detail);
  if (detail && typeof detail === "object" && "message" in detail) {
    return new Error(String((detail as { message: unknown }).message));
  }
  return new Error(fallback);
}

async function postBatchSelection<T>(
  session: string,
  action: "review" | "commit",
  pairs: BatchPairSelection[],
  singles: BatchSingleSelection[],
  physicalOutlines: Record<string, Poly> = {},
  readyOnly = false,
  discardBlocked = false,
): Promise<T> {
  const r = await fetch(`/api/batch/${session}/${action}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      pairs, singles, physical_outlines: physicalOutlines,
      ready_only: readyOnly, discard_blocked: discardBlocked,
    }),
  });
  const data = await r.json().catch(() => ({ detail: r.statusText }));
  if (!r.ok) throw batchError(data.detail, `batch ${action} failed`);
  return data as T;
}

export function reviewBatch(
  session: string,
  pairs: BatchPairSelection[],
  singles: BatchSingleSelection[],
  physicalOutlines: Record<string, Poly> = {},
): Promise<BatchReviewResult> {
  return postBatchSelection(session, "review", pairs, singles, physicalOutlines);
}

export function commitBatch(
  session: string,
  pairs: BatchPairSelection[],
  singles: BatchSingleSelection[],
  physicalOutlines: Record<string, Poly> = {},
  readyOnly = false,
  discardBlocked = false,
): Promise<BatchCommitResult> {
  return postBatchSelection(
    session, "commit", pairs, singles, physicalOutlines, readyOnly, discardBlocked,
  );
}

export interface LibraryTool {
  id: string;
  label: string;
  grid_x: number;
  grid_y: number;
  thickness_mm: number;
  silhouette_height_mm: number;
  full_height_mm: number | null;
  clearance_mm: number;
  bin_style: BinStyle;
  pocket_depth_mm: number | null;
  derived_pocket_depth_mm: number | null;
  derived_height_u: number | null;
  derived_overall_height_mm: number | null;
  derived_key: string | null;
  derived_reserved_cells: [number, number][];
  derived_available_cells: [number, number][];
  lip: boolean;
  round_tool: boolean;
  finger_hole: boolean;
  has_photo: boolean;
  source_project: string;
  source_tool: string;
  created_ts: number;
  thumb: string;
  photo_thumb: string | null;
  readiness: ReadinessReport;
  provenance: ArtifactProvenance | null;
  outline_revision: number;
}

export interface PhotoOutline {
  has_photo: boolean;
  display?: string;
  width?: number;
  height?: number;
  outline: [number, number][];
}

export async function getLibraryPhotoOutline(id: string): Promise<PhotoOutline> {
  const r = await fetch(`/api/library/${id}/photo-outline`, { cache: "no-store" });
  if (!r.ok) throw new Error("photo-outline fetch failed");
  return r.json();
}

export async function saveLibraryOutlinePx(
  id: string,
  points: [number, number][],
): Promise<LibraryTool> {
  const r = await fetch(`/api/library/${id}/outline-px`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ points }),
  });
  if (!r.ok) {
    const d = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(d.detail ?? "save failed");
  }
  return r.json();
}

export interface ComposeResult {
  drawer: { cols: number; rows: number };
  tools: LibraryTool[];
  layout: {
    placed: DrawerPlacement[];
    overflow: string[];
    used_cols: number;
    used_rows: number;
  };
}

export async function listLibrary(): Promise<LibraryTool[]> {
  const r = await fetch("/api/library");
  if (!r.ok) throw new Error("library list failed");
  return (await r.json()).tools;
}

export async function downloadLibraryArchive(): Promise<void> {
  const r = await fetch("/api/library-export.zip", { cache: "no-store" });
  if (!r.ok) {
    const d = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(d.detail ?? "library export failed");
  }
  const url = URL.createObjectURL(await r.blob());
  const a = document.createElement("a");
  a.href = url;
  a.download = "gridshot-library.zip";
  a.click();
  URL.revokeObjectURL(url);
}

export interface LibraryBackupResult {
  filename: string;
  bytes: number;
  tool_count: number;
  created_at: string;
}

export async function createLibraryBackup(): Promise<LibraryBackupResult> {
  const r = await fetch("/api/library-backups", { method: "POST" });
  const data = await r.json().catch(() => ({ detail: r.statusText }));
  if (!r.ok) throw new Error(data.detail ?? "library backup failed");
  return data as LibraryBackupResult;
}

export async function addToLibrary(project: string): Promise<number> {
  const r = await fetch(`/api/library/add/${project}`, { method: "POST" });
  if (!r.ok) {
    const d = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(d.detail ?? "save to library failed");
  }
  return (await r.json()).added.length;
}

export async function deleteLibraryTool(id: string): Promise<void> {
  await fetch(`/api/library/${id}`, { method: "DELETE" });
}

export interface LibraryEditResult extends OutlineEditSession {
  raw: Poly;
  corrected: Poly;
  grid_x: number;
  grid_y: number;
}

export async function libraryEditStart(id: string): Promise<LibraryEditResult> {
  const r = await fetch(`/api/library/${id}/edit`, { method: "POST" });
  if (!r.ok) {
    const d = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(d.detail ?? "no photo for this tool");
  }
  return r.json();
}

export async function libraryEditClick(
  session: string,
  points: ClickPoint[],
  box?: [number, number, number, number] | null,
): Promise<LibraryEditResult> {
  return editorPrompt(`/api/library/edit/${session}/click`, points, box);
}

async function editorPrompt<T extends OutlineEditState>(
  url: string,
  points: ClickPoint[],
  box?: [number, number, number, number] | null,
): Promise<T> {
  const fd = new FormData();
  fd.append("points", JSON.stringify(points.map((p) => [p.x, p.y])));
  fd.append("labels", JSON.stringify(points.map((p) => p.label)));
  if (box) fd.append("box", JSON.stringify(box));
  const r = await fetch(url, { method: "POST", body: fd });
  if (!r.ok) {
    const d = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(d.detail ?? "segment failed");
  }
  return r.json();
}

async function editorJson<T extends OutlineEditState>(
  url: string,
  polygon?: Poly,
): Promise<T> {
  const r = await fetch(url, {
    method: "POST",
    headers: polygon ? { "Content-Type": "application/json" } : undefined,
    body: polygon ? JSON.stringify(polygon) : undefined,
  });
  if (!r.ok) {
    const d = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(d.detail ?? "outline edit failed");
  }
  return r.json();
}

export function libraryEditOutline(
  session: string,
  polygon: Poly,
): Promise<LibraryEditResult> {
  return editorJson(`/api/library/edit/${session}/outline`, polygon);
}

export function libraryEditHistory(
  session: string,
  direction: "undo" | "redo",
): Promise<LibraryEditResult> {
  return editorJson(`/api/library/edit/${session}/history/${direction}`);
}

export async function libraryEditSave(
  session: string,
  outlineVariant: OutlineVariant | "recommended" = "recommended",
): Promise<LibraryTool> {
  const body = new FormData();
  body.append("outline_variant", outlineVariant);
  const r = await fetch(`/api/library/edit/${session}/save`, {
    method: "POST",
    body,
  });
  if (!r.ok) {
    const d = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(d.detail ?? "library outline save failed");
  }
  return r.json();
}

export async function getLibraryOutline(id: string): Promise<Poly | null> {
  const r = await fetch(`/api/library/${id}/outline`, { cache: "no-store" });
  if (!r.ok) throw new Error("outline fetch failed");
  return (await r.json()).outline;
}

export async function updateLibraryTool(
  id: string,
  changes: {
    label?: string;
    thickness_mm?: number;
    silhouette_height_mm?: number;
    full_height_mm?: number | null;
    clearance_mm?: number;
    bin_style?: BinStyle;
    pocket_depth_mm?: number | null;
    round_tool?: boolean;
    finger_hole?: boolean;
    outline?: Poly;
    raw_outline?: Poly;
    edit_source?: "sam" | "manual" | "physical";
    edit_diagnostics?: Record<string, number>;
  },
): Promise<LibraryTool> {
  const r = await fetch(`/api/library/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(changes),
  });
  if (!r.ok) throw new Error("update failed");
  return r.json();
}

export async function composeLibrary(
  ids: string[],
  cols: number,
  rows: number,
  overallHeight?: number | null,
): Promise<ComposeResult> {
  const r = await fetch("/api/library/compose", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids, cols, rows, overall_height: overallHeight ?? null }),
  });
  if (!r.ok) throw new Error("compose failed");
  return r.json();
}

export async function drawerPreviewGlb(
  ids: string[],
  cols: number,
  rows: number,
  overallHeight?: number | null,
): Promise<Blob> {
  const r = await fetch("/api/library/compose/preview.glb", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids, cols, rows, overall_height: overallHeight ?? null }),
  });
  if (!r.ok) {
    const detail = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(detail.detail ?? "drawer preview failed");
  }
  return r.blob();
}

export interface Placement {
  id: string;
  tx: number;
  ty: number;
  rot: number;
}

export interface CombineToolOverride {
  id: string;
  finger_hole: boolean | null;
}

export interface CombineTool extends Placement {
  label: string;
  bin_style: BinStyle;
  depth_mm: number;
  depth_mode: "automatic" | "library override";
  clearance_mm: number;
  round_tool: boolean;
  finger: boolean;
  finger_hole: boolean;
  finger_hole_inherited: boolean;
  finger_hole_override: boolean | null;
  finger_holes: [number, number, number][];
  stamp: [number, number][];
}

export interface CombinePreview {
  bin_style: BinStyle;
  gx: number;
  gy: number;
  outer_w: number;
  outer_d: number;
  overall_height_mm: number;
  pitch: number;
  bin_size: number;
  wall: number;
  lip: boolean;
  reserved_cells: [number, number][];
  available_cells: [number, number][];
  tools: CombineTool[];
}

export async function combinePreview(
  ids: string[],
  placements?: Placement[] | null,
  overallHeight?: number | null,
  lip = true,
  overrides?: CombineToolOverride[] | null,
  binStyle: BinStyle = "pocket",
): Promise<CombinePreview> {
  const r = await fetch("/api/library/combine/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids, placements: placements ?? null, overall_height: overallHeight ?? null, lip, overrides: overrides ?? null, bin_style: binStyle }),
  });
  if (!r.ok) {
    const d = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(d.detail ?? "preview failed");
  }
  return r.json();
}

export async function combinePreviewGlb(
  ids: string[],
  placements?: Placement[] | null,
  overallHeight?: number | null,
  lip = true,
  overrides?: CombineToolOverride[] | null,
  binStyle: BinStyle = "pocket",
): Promise<Blob> {
  const r = await fetch("/api/library/combine/preview.glb", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids, placements: placements ?? null, overall_height: overallHeight ?? null, lip, overrides: overrides ?? null, bin_style: binStyle }),
  });
  if (!r.ok) {
    const d = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(d.detail ?? "3D preview failed");
  }
  return r.blob();
}

export async function combineLibrary(
  ids: string[],
  placements?: Placement[] | null,
  overallHeight?: number | null,
  lip = true,
  overrides?: CombineToolOverride[] | null,
  binStyle: BinStyle = "pocket",
): Promise<void> {
  const r = await fetch("/api/library/combine", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids, placements: placements ?? null, overall_height: overallHeight ?? null, lip, overrides: overrides ?? null, bin_style: binStyle }),
  });
  if (!r.ok) {
    const d = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(d.detail ?? "combine failed");
  }
  const url = URL.createObjectURL(await r.blob());
  const a = document.createElement("a");
  a.href = url;
  a.download = "multitool-bin.3mf";
  a.click();
  URL.revokeObjectURL(url);
}

export async function exportDrawer(
  ids: string[],
  cols: number,
  rows: number,
  overallHeight?: number | null,
): Promise<void> {
  const r = await fetch("/api/library/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids, cols, rows, overall_height: overallHeight ?? null }),
  });
  if (!r.ok) throw new Error("export failed");
  const url = URL.createObjectURL(await r.blob());
  const a = document.createElement("a");
  a.href = url;
  a.download = "drawer.zip";
  a.click();
  URL.revokeObjectURL(url);
}

export interface GenerateParams {
  thickness?: number | null;
  full_height?: number | null;
  clearance: number;
  bin_style: BinStyle;
  depth?: number | null;
  overall_height?: number | null;
  finger_hole: boolean;
  lip: boolean;
  round_tool?: boolean;
}

export async function startSession(
  file: File,
  file2?: File | null,
): Promise<Session> {
  const fd = new FormData();
  fd.append("file", file);
  if (file2) fd.append("file2", file2);
  const r = await fetch("/api/session", { method: "POST", body: fd });
  if (!r.ok) {
    const d = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(d.detail ?? "session failed");
  }
  return r.json();
}
export async function getSession(session: string): Promise<Session> {
  const r = await fetch(`/api/session/${session}`, { cache: "no-store" });
  if (!r.ok) {
    const d = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(d.detail ?? "session failed to load");
  }
  return r.json();
}


export interface SessionEditResult extends OutlineEditState {
  warnings: string[];
  readiness: ReadinessReport;
}

export async function sessionClick(
  session: string,
  points: ClickPoint[],
  box?: [number, number, number, number] | null,
): Promise<SessionEditResult> {
  return editorPrompt(`/api/session/${session}/click`, points, box);
}

export function sessionSetOutline(
  session: string,
  polygon: Poly,
): Promise<SessionEditResult> {
  return editorJson(`/api/session/${session}/outline`, polygon);
}

export function sessionEditHistory(
  session: string,
  direction: "undo" | "redo",
): Promise<SessionEditResult> {
  return editorJson(`/api/session/${session}/history/${direction}`);
}

export interface PhysicalOutlineEditResult {
  polygon: Poly;
  revision: number;
  diagnostics: FootprintReconstruction;
}

export async function sessionSetPhysicalOutline(
  session: string,
  polygon: Poly,
): Promise<PhysicalOutlineEditResult> {
  const r = await fetch(`/api/session/${session}/physical-outline`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(polygon),
  });
  if (!r.ok) {
    const d = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(d.detail ?? "physical cutout save failed");
  }
  return r.json();
}

export async function sessionAddToLibrary(
  session: string,
  params: GenerateParams,
  outlineVariant: OutlineVariant,
): Promise<LibraryTool> {
  const fd = new FormData();
  if (params.thickness != null) fd.append("thickness", String(params.thickness));
  if (params.full_height != null) fd.append("full_height", String(params.full_height));
  fd.append("clearance", String(params.clearance));
  fd.append("bin_style", params.bin_style);
  if (params.depth != null) fd.append("depth", String(params.depth));
  fd.append("finger_hole", String(params.finger_hole));
  fd.append("lip", String(params.lip));
  fd.append("round_tool", String(params.round_tool ?? false));
  fd.append("outline_variant", outlineVariant);
  const r = await fetch(`/api/session/${session}/library`, { method: "POST", body: fd });
  if (!r.ok) {
    const d = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(d.detail ?? "save to library failed");
  }
  return r.json();
}

export async function sessionGenerate(
  session: string,
  params: GenerateParams,
  outlineVariant: OutlineVariant | "recommended" = "recommended",
): Promise<TraceResult> {
  const fd = new FormData();
  if (params.thickness != null) fd.append("thickness", String(params.thickness));
  if (params.full_height != null) fd.append("full_height", String(params.full_height));
  fd.append("clearance", String(params.clearance));
  fd.append("bin_style", params.bin_style);
  if (params.depth != null) fd.append("depth", String(params.depth));
  if (params.overall_height != null)
    fd.append("overall_height", String(params.overall_height));
  fd.append("finger_hole", String(params.finger_hole));
  fd.append("lip", String(params.lip));
  fd.append("round_tool", String(params.round_tool ?? false));
  fd.append("outline_variant", outlineVariant);
  const r = await fetch(`/api/session/${session}/generate`, { method: "POST", body: fd });
  if (!r.ok) {
    const d = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(d.detail ?? "generate failed");
  }
  return r.json();
}

export interface TraceParams {
  file2?: File | null;
  thickness?: number | null;
  full_height?: number | null;
  clearance: number;
  bin_style: BinStyle;
  depth?: number | null;
  overall_height?: number | null;
  finger_hole: boolean;
  lip: boolean;
  round_tool?: boolean;
  mat_id?: string | null;
}

export async function postTrace(
  file: File,
  params: TraceParams,
): Promise<TraceResult> {
  const fd = new FormData();
  fd.append("file", file);
  if (params.file2) fd.append("file2", params.file2);
  if (params.thickness != null) fd.append("thickness", String(params.thickness));
  if (params.full_height != null) fd.append("full_height", String(params.full_height));
  fd.append("clearance", String(params.clearance));
  fd.append("bin_style", params.bin_style);
  if (params.depth != null) fd.append("depth", String(params.depth));
  if (params.overall_height != null)
    fd.append("overall_height", String(params.overall_height));
  fd.append("finger_hole", String(params.finger_hole));
  fd.append("lip", String(params.lip));
  fd.append("round_tool", String(params.round_tool ?? false));
  if (params.mat_id) fd.append("mat_id", params.mat_id);

  const r = await fetch("/api/trace", { method: "POST", body: fd });
  if (!r.ok) {
    const detail = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(detail.detail ?? "trace failed");
  }
  return r.json();
}
