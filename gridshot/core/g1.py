"""Physical G1 accuracy harness.

The harness keeps accuracy evidence as source data, not screenshots or prose.
A run manifest owns the capture matrix, source/calibration provenance, every
outline stage, physical truth, fit outcomes, and the release-gate report.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from shapely.affinity import rotate as rotate_shape
from shapely.affinity import translate as translate_shape
from shapely.geometry import LineString, Point

from . import bench as bench_mod
from . import contour as contour_mod
from . import derive as derive_mod
from .models import Calibration, Poly, PrinterProfile

SCHEMA_VERSION = "g1.v1"
PROTOCOL_VERSION = "2026-07-21"
FLAT_ERROR_LIMIT_MM = 0.3
THICK_ERROR_LIMIT_MM = 0.5
FIRST_PRINT_FIT_MIN = 0.90
RECAPTURE_RATE_MAX = 0.10

ToolClass = Literal["flat", "thick", "glossy", "dark", "asymmetric"]
Position = Literal["center", "edge"]
Lighting = Literal["normal", "challenging"]
SampleStatus = Literal["planned", "captured", "measured"]
FitOutcome = Literal["untested", "fit", "too_tight", "too_loose"]
GateState = Literal["pass", "fail", "incomplete"]

RELEASE_TOOL_CLASSES: tuple[ToolClass, ...] = (
    "flat",
    "thick",
    "glossy",
    "dark",
    "asymmetric",
)


class G1Environment(BaseModel):
    operator: str = ""
    printer: str = ""
    material: str = ""
    nozzle_mm: float | None = None
    slicer_profile: str = ""
    notes: str = ""


class G1Condition(BaseModel):
    tool_class: ToolClass
    position: Position
    lighting: Lighting = "normal"


class G1Provenance(BaseModel):
    source_result: str = ""
    source_images: list[str] = Field(default_factory=list)
    calibration: Calibration | None = None
    mat_id: str | None = None
    device_profile_id: str | None = None
    printer_profile: PrinterProfile | None = None
    derivation_key: str | None = None


class G1Geometry(BaseModel):
    raw_outline: Poly | None = None
    corrected_outline: Poly | None = None
    compensated_outline: Poly | None = None
    truth_outline: Poly | None = None
    truth_source: str = ""
    estimated_thickness_mm: float | None = Field(None, gt=0)
    truth_thickness_mm: float | None = Field(None, gt=0)
    clearance_mm: float = Field(1.0, ge=0)
    camera_height_mm: float | None = Field(None, gt=0)
    camera_nadir_xy_mm: tuple[float, float] | None = None
    tool_center_xy_mm: tuple[float, float] | None = None


class G1Outcome(BaseModel):
    fit: FitOutcome = "untested"
    corrections: int | None = Field(None, ge=0)
    recaptures: int | None = Field(None, ge=0)
    notes: str = ""


class G1Sample(BaseModel):
    id: str
    tool_id: str = ""
    required: bool = True
    condition: G1Condition
    status: SampleStatus = "planned"
    provenance: G1Provenance = Field(default_factory=G1Provenance)
    geometry: G1Geometry = Field(default_factory=G1Geometry)
    outcome: G1Outcome = Field(default_factory=G1Outcome)


class G1Manifest(BaseModel):
    schema_version: Literal["g1.v1"] = SCHEMA_VERSION
    protocol_version: str = PROTOCOL_VERSION
    run_id: str
    created_at: str
    environment: G1Environment = Field(default_factory=G1Environment)
    samples: list[G1Sample] = Field(default_factory=list)


def default_samples() -> list[G1Sample]:
    """Required release matrix: five tool classes at mat center and edge."""
    samples: list[G1Sample] = []
    for tool_class in RELEASE_TOOL_CLASSES:
        for position in ("center", "edge"):
            samples.append(
                G1Sample(
                    id=f"{tool_class}-{position}",
                    condition=G1Condition(
                        tool_class=tool_class,
                        position=position,
                        lighting=(
                            "challenging"
                            if tool_class in {"glossy", "dark"}
                            else "normal"
                        ),
                    ),
                )
            )
    return samples


def new_manifest(
    run_id: str,
    *,
    environment: G1Environment | None = None,
) -> G1Manifest:
    return G1Manifest(
        run_id=run_id,
        created_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        environment=environment or G1Environment(),
        samples=default_samples(),
    )


def load_manifest(path: Path) -> G1Manifest:
    return G1Manifest.model_validate_json(path.read_text())


def save_manifest(manifest: G1Manifest, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2))
    return path


def sample_by_id(manifest: G1Manifest, sample_id: str) -> G1Sample:
    for sample in manifest.samples:
        if sample.id == sample_id:
            return sample
    raise KeyError(sample_id)


def load_poly(path: Path) -> Poly:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "poly" in payload:
        payload = payload["poly"]
    return Poly.model_validate(payload)


def ingest_result(
    manifest: G1Manifest,
    sample_id: str,
    result: dict,
    *,
    source_result: str = "",
    tool_id: str | None = None,
    printer_profile: PrinterProfile | None = None,
) -> G1Sample:
    """Persist every geometry stage from a trace result into one matrix sample."""
    sample = sample_by_id(manifest, sample_id)
    corrected_data = result.get("corrected_poly")
    if corrected_data is None:
        raise ValueError("trace result has no corrected_poly")
    corrected = Poly.model_validate(corrected_data)
    raw = Poly.model_validate(result.get("raw_poly") or corrected_data)
    bin_data = result.get("bin") or {}
    thickness = bin_data.get("thickness_mm")
    if thickness is None or float(thickness) <= 0:
        raise ValueError("trace result has no positive thickness")
    printer = (
        printer_profile
        or bench_mod.load_profile()
        or bench_mod.default_profile()
    )
    settings = derive_mod.BinSettings(
        clearance_mm=float(bin_data.get("clearance_mm", 1.0)),
        pocket_depth_mm=bin_data.get("pocket_depth_override_mm"),
        lip=bool(bin_data.get("lip", True)),
        finger_hole=bool(bin_data.get("finger_hole", False)),
        round_tool=bool(bin_data.get("round_tool", False)),
    )
    derived = derive_mod.derive_bin_spec(
        derive_mod.ToolGeometry(
            outline=corrected,
            thickness_mm=float(thickness),
        ),
        settings,
        printer,
    )

    calibration_data = result.get("calibration_model")
    calibration = (
        Calibration.model_validate(calibration_data)
        if calibration_data is not None
        else None
    )
    calibration_summary = result.get("calibration") or {}
    provenance_data = result.get("provenance") or {}
    source_images = list(provenance_data.get("source_images") or [])
    center = contour_mod.to_shapely(corrected).centroid
    nadir = (
        calibration.nadir_xy_mm
        if calibration is not None
        else calibration_summary.get("nadir_xy_mm")
    )

    sample.tool_id = tool_id if tool_id is not None else sample.tool_id
    sample.status = "captured"
    sample.provenance = G1Provenance(
        source_result=source_result,
        source_images=source_images,
        calibration=calibration,
        mat_id=(
            calibration.mat_id
            if calibration is not None
            else provenance_data.get("mat_id")
        ),
        device_profile_id=(
            calibration.device_profile_id
            if calibration is not None
            else provenance_data.get("device_profile_id")
        ),
        printer_profile=printer,
        derivation_key=derived.derivation_key,
    )
    sample.geometry = sample.geometry.model_copy(
        update={
            "raw_outline": raw,
            "corrected_outline": corrected,
            "compensated_outline": derived.compensated_poly,
            "estimated_thickness_mm": float(thickness),
            "clearance_mm": settings.clearance_mm,
            "camera_height_mm": (
                calibration.camera_height_mm
                if calibration is not None
                else calibration_summary.get("camera_height_mm")
            ),
            "camera_nadir_xy_mm": tuple(nadir) if nadir is not None else None,
            "tool_center_xy_mm": (float(center.x), float(center.y)),
        }
    )
    return sample


def record_sample(
    manifest: G1Manifest,
    sample_id: str,
    *,
    truth_outline: Poly | None = None,
    truth_source: str | None = None,
    truth_thickness_mm: float | None = None,
    fit: FitOutcome | None = None,
    corrections: int | None = None,
    recaptures: int | None = None,
    notes: str | None = None,
) -> G1Sample:
    sample = sample_by_id(manifest, sample_id)
    geometry = sample.geometry.model_dump()
    if truth_outline is not None:
        geometry["truth_outline"] = truth_outline
    if truth_source is not None:
        geometry["truth_source"] = truth_source
    if truth_thickness_mm is not None:
        geometry["truth_thickness_mm"] = truth_thickness_mm
    sample.geometry = G1Geometry.model_validate(geometry)

    outcome = sample.outcome.model_dump()
    if fit is not None:
        outcome["fit"] = fit
    if corrections is not None:
        outcome["corrections"] = corrections
    if recaptures is not None:
        outcome["recaptures"] = recaptures
    if notes is not None:
        outcome["notes"] = notes
    sample.outcome = G1Outcome.model_validate(outcome)

    if _measurement_complete(sample):
        sample.status = "measured"
    return sample


def _ring_points(ring: list[tuple[float, float]], spacing_mm: float) -> np.ndarray:
    closed = [*ring, ring[0]]
    line = LineString(closed)
    count = max(16, min(2048, math.ceil(line.length / spacing_mm)))
    distances = np.linspace(0.0, line.length, count, endpoint=False)
    return np.asarray(
        [(line.interpolate(float(distance)).x, line.interpolate(float(distance)).y)
         for distance in distances],
        dtype=np.float64,
    )


def boundary_points(poly: Poly, spacing_mm: float = 0.25) -> np.ndarray:
    rings = [poly.exterior, *poly.holes]
    return np.vstack([_ring_points(ring, spacing_mm) for ring in rings])


def _rotate_points(points: np.ndarray, angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    return points @ rotation.T


def register_outline(
    predicted: Poly,
    truth: Poly,
) -> tuple[Poly, dict[str, float]]:
    """Rigidly register predicted to truth; scale is deliberately not fitted."""
    if predicted == truth:
        return predicted, {
            "rotation_deg": 0.0,
            "translation_x_mm": 0.0,
            "translation_y_mm": 0.0,
        }
    predicted_shape = contour_mod.to_shapely(predicted).buffer(0)
    truth_shape = contour_mod.to_shapely(truth).buffer(0)
    if predicted_shape.is_empty or truth_shape.is_empty:
        raise ValueError("cannot register an empty outline")

    predicted_points = boundary_points(predicted)
    truth_points = boundary_points(truth)
    predicted_centroid = np.asarray(predicted_shape.centroid.coords[0])
    truth_centroid = np.asarray(truth_shape.centroid.coords[0])
    predicted_centered = predicted_points - predicted_centroid
    truth_centered = truth_points - truth_centroid
    truth_tree = cKDTree(truth_centered)

    def objective(params: np.ndarray) -> float:
        angle, tx, ty = params
        moved = _rotate_points(predicted_centered, float(angle))
        moved += np.asarray([tx, ty])
        forward = truth_tree.query(moved, k=1)[0]
        reverse = cKDTree(moved).query(truth_centered, k=1)[0]
        return float(np.mean(np.concatenate([forward, reverse])))

    coarse_angles = np.arange(-180.0, 180.0, 5.0)
    coarse_scores = [
        objective(np.asarray([angle, 0.0, 0.0]))
        for angle in coarse_angles
    ]
    initial_angle = float(coarse_angles[int(np.argmin(coarse_scores))])
    solved = minimize(
        objective,
        np.asarray([initial_angle, 0.0, 0.0]),
        method="Powell",
        options={"xtol": 1e-5, "ftol": 1e-6, "maxiter": 300},
    )
    angle, tx, ty = (float(value) for value in solved.x)
    aligned_shape = rotate_shape(
        predicted_shape,
        angle,
        origin=tuple(predicted_centroid),
        use_radians=False,
    )
    aligned_shape = translate_shape(
        aligned_shape,
        xoff=float(truth_centroid[0] - predicted_centroid[0] + tx),
        yoff=float(truth_centroid[1] - predicted_centroid[1] + ty),
    )
    return contour_mod.from_shapely(aligned_shape), {
        "rotation_deg": angle,
        "translation_x_mm": float(
            truth_centroid[0] - predicted_centroid[0] + tx
        ),
        "translation_y_mm": float(
            truth_centroid[1] - predicted_centroid[1] + ty
        ),
    }


def _boundary_evaluation(
    predicted: Poly,
    truth: Poly,
) -> tuple[dict, np.ndarray, Poly]:
    aligned, registration = register_outline(predicted, truth)
    predicted_points = boundary_points(aligned)
    truth_points = boundary_points(truth)
    forward = cKDTree(truth_points).query(predicted_points, k=1)[0]
    reverse = cKDTree(predicted_points).query(truth_points, k=1)[0]
    absolute = np.concatenate([forward, reverse])

    truth_shape = contour_mod.to_shapely(truth).buffer(0)
    signed = np.asarray(
        [
            -distance if truth_shape.contains(Point(point)) else distance
            for point, distance in zip(predicted_points, forward)
        ]
    )
    metrics = {
        "p50_abs_mm": float(np.percentile(absolute, 50)),
        "p95_abs_mm": float(np.percentile(absolute, 95)),
        "max_abs_mm": float(np.max(absolute)),
        "mean_signed_bias_mm": float(np.mean(signed)),
        "registration": registration,
    }
    return metrics, absolute, aligned


def boundary_metrics(predicted: Poly, truth: Poly) -> dict:
    return _boundary_evaluation(predicted, truth)[0]


def sample_metrics(sample: G1Sample) -> tuple[dict, np.ndarray]:
    geometry = sample.geometry
    if geometry.corrected_outline is None or geometry.truth_outline is None:
        raise ValueError(f"sample {sample.id} has no corrected/truth outline pair")
    boundary, distances, _ = _boundary_evaluation(
        geometry.corrected_outline,
        geometry.truth_outline,
    )
    thickness_error = None
    if (
        geometry.estimated_thickness_mm is not None
        and geometry.truth_thickness_mm is not None
    ):
        thickness_error = (
            geometry.estimated_thickness_mm - geometry.truth_thickness_mm
        )
    parallax_term = None
    if (
        geometry.tool_center_xy_mm is not None
        and geometry.camera_nadir_xy_mm is not None
        and geometry.camera_height_mm is not None
        and geometry.truth_thickness_mm is not None
    ):
        radius = math.dist(
            geometry.tool_center_xy_mm,
            geometry.camera_nadir_xy_mm,
        )
        parallax_term = (
            radius
            * geometry.truth_thickness_mm
            / geometry.camera_height_mm
        )
    return {
        "id": sample.id,
        "tool_id": sample.tool_id,
        "tool_class": sample.condition.tool_class,
        "position": sample.condition.position,
        "lighting": sample.condition.lighting,
        "boundary": boundary,
        "thickness_error_mm": thickness_error,
        "parallax_term_mm": parallax_term,
        "fit": sample.outcome.fit,
        "corrections": sample.outcome.corrections,
        "recaptures": sample.outcome.recaptures,
    }, distances


def _distribution(values: list[float] | np.ndarray) -> dict | None:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return None
    return {
        "count": int(array.size),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _gate(
    state: GateState,
    *,
    value=None,
    limit=None,
    message: str,
) -> dict:
    return {
        "state": state,
        "value": value,
        "limit": limit,
        "message": message,
    }


def _accuracy_gate(
    values: list[float],
    limit: float,
    complete: bool,
    label: str,
) -> dict:
    distribution = _distribution(values)
    if distribution is None:
        return _gate(
            "incomplete",
            limit=limit,
            message=f"no measured {label} boundary samples",
        )
    value = distribution["p95"]
    if value > limit:
        state: GateState = "fail"
    elif complete:
        state = "pass"
    else:
        state = "incomplete"
    return _gate(
        state,
        value=value,
        limit=limit,
        message=f"{label} boundary p95 {value:.3f}mm",
    )


def _measurement_complete(sample: G1Sample) -> bool:
    geometry = sample.geometry
    return (
        geometry.corrected_outline is not None
        and geometry.truth_outline is not None
        and geometry.estimated_thickness_mm is not None
        and geometry.truth_thickness_mm is not None
    )


def _provenance_complete(sample: G1Sample) -> bool:
    provenance = sample.provenance
    geometry = sample.geometry
    return bool(
        sample.tool_id
        and provenance.source_result
        and provenance.source_images
        and provenance.calibration is not None
        and provenance.printer_profile is not None
        and provenance.derivation_key
        and geometry.raw_outline is not None
        and geometry.compensated_outline is not None
        and geometry.truth_source
        and sample.outcome.corrections is not None
        and sample.outcome.recaptures is not None
        and _measurement_complete(sample)
    )


def build_report(manifest: G1Manifest) -> dict:
    required = [sample for sample in manifest.samples if sample.required]
    measured = [
        sample
        for sample in required
        if sample.status == "measured" and _measurement_complete(sample)
    ]
    metrics: list[dict] = []
    distance_by_sample: dict[str, np.ndarray] = {}
    for sample in measured:
        metric, distances = sample_metrics(sample)
        metrics.append(metric)
        distance_by_sample[sample.id] = distances

    matrix_complete = len(measured) == len(required) and bool(required)
    flat_values: list[float] = []
    thick_values: list[float] = []
    for sample in measured:
        values = distance_by_sample[sample.id].tolist()
        if (sample.geometry.truth_thickness_mm or 0.0) <= 2.0:
            flat_values.extend(values)
        else:
            thick_values.extend(values)

    groups: dict[str, dict] = {}
    for dimension, getter in {
        "class": lambda sample: sample.condition.tool_class,
        "position": lambda sample: sample.condition.position,
    }.items():
        for sample in measured:
            key = f"{dimension}:{getter(sample)}"
            group = groups.setdefault(
                key,
                {
                    "boundary_values": [],
                    "thickness_errors": [],
                    "samples": 0,
                    "corrections": 0,
                    "recaptures": 0,
                },
            )
            group["boundary_values"].extend(
                distance_by_sample[sample.id].tolist()
            )
            if (
                sample.geometry.estimated_thickness_mm is not None
                and sample.geometry.truth_thickness_mm is not None
            ):
                group["thickness_errors"].append(
                    abs(
                        sample.geometry.estimated_thickness_mm
                        - sample.geometry.truth_thickness_mm
                    )
                )
            group["samples"] += 1
            group["corrections"] += sample.outcome.corrections or 0
            group["recaptures"] += sample.outcome.recaptures or 0
    for group in groups.values():
        group["boundary_error_mm"] = _distribution(
            group.pop("boundary_values")
        )
        group["thickness_error_mm"] = _distribution(
            group.pop("thickness_errors")
        )

    default_clearance = [
        sample
        for sample in required
        if abs(sample.geometry.clearance_mm - 1.0) < 1e-9
        and sample.outcome.fit != "untested"
    ]
    fit_complete = len(default_clearance) == len(required) and {
        sample.id for sample in default_clearance
    } == {sample.id for sample in required}
    fit_count = sum(
        sample.outcome.fit == "fit"
        for sample in default_clearance
    )
    fit_rate = (
        fit_count / len(default_clearance)
        if default_clearance
        else None
    )
    too_tight = sum(
        sample.outcome.fit == "too_tight"
        for sample in default_clearance
    )
    recapture_complete = bool(measured) and all(
        sample.outcome.recaptures is not None for sample in measured
    )
    recapture_rate = (
        sum((sample.outcome.recaptures or 0) > 0 for sample in measured)
        / len(measured)
        if recapture_complete
        else None
    )

    parallax_pairs = [
        (metric["parallax_term_mm"], metric["boundary"]["mean_signed_bias_mm"])
        for metric in metrics
        if metric["parallax_term_mm"] is not None
    ]
    parallax_fit = None
    if len(parallax_pairs) >= 2:
        x = np.asarray([pair[0] for pair in parallax_pairs])
        y = np.asarray([pair[1] for pair in parallax_pairs])
        slope, intercept = np.polyfit(x, y, 1)
        parallax_fit = {
            "samples": len(parallax_pairs),
            "intercept_mm": float(intercept),
            "slope": float(slope),
        }

    gates = {
        "matrix_complete": _gate(
            "pass" if matrix_complete else "incomplete",
            value=f"{len(measured)}/{len(required)}",
            limit=f"{len(required)}/{len(required)}",
            message="required capture conditions with registered physical truth",
        ),
        "flat_outline_accuracy": _accuracy_gate(
            flat_values,
            FLAT_ERROR_LIMIT_MM,
            matrix_complete,
            "flat",
        ),
        "thick_outline_accuracy": _accuracy_gate(
            thick_values,
            THICK_ERROR_LIMIT_MM,
            matrix_complete,
            "thick/non-flat",
        ),
    }
    if fit_rate is None:
        gates["first_print_fit"] = _gate(
            "incomplete",
            limit=FIRST_PRINT_FIT_MIN,
            message="no default-clearance fit trials recorded",
        )
    else:
        fit_state: GateState
        if fit_rate < FIRST_PRINT_FIT_MIN:
            fit_state = "fail"
        elif fit_complete:
            fit_state = "pass"
        else:
            fit_state = "incomplete"
        gates["first_print_fit"] = _gate(
            fit_state,
            value=fit_rate,
            limit=FIRST_PRINT_FIT_MIN,
            message=(
                f"{fit_count}/{len(default_clearance)} default-clearance outputs fit"
            ),
        )
    gates["too_tight_failures"] = _gate(
        (
            "fail"
            if too_tight
            else ("pass" if fit_complete else "incomplete")
        ),
        value=too_tight,
        limit=0,
        message="default-clearance outputs that did not accept the tool",
    )
    if recapture_rate is None:
        gates["recapture_burden"] = _gate(
            "incomplete",
            limit=RECAPTURE_RATE_MAX,
            message="recapture counts are incomplete",
        )
    else:
        gates["recapture_burden"] = _gate(
            (
                "fail"
                if recapture_rate >= RECAPTURE_RATE_MAX
                else ("pass" if matrix_complete else "incomplete")
            ),
            value=recapture_rate,
            limit=RECAPTURE_RATE_MAX,
            message="share of capture conditions requiring another photo",
        )
    provenance_ok = bool(required) and all(
        _provenance_complete(sample) for sample in measured
    )
    gates["provenance"] = _gate(
        (
            "pass"
            if matrix_complete and provenance_ok
            else ("fail" if matrix_complete else "incomplete")
        ),
        value=sum(_provenance_complete(sample) for sample in measured),
        limit=len(required),
        message="measured samples reproducible from persisted sources",
    )

    states = [gate["state"] for gate in gates.values()]
    overall: GateState = (
        "fail"
        if "fail" in states
        else ("incomplete" if "incomplete" in states else "pass")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": manifest.protocol_version,
        "run_created_at": manifest.created_at,
        "environment": manifest.environment.model_dump(mode="json"),
        "run_id": manifest.run_id,
        "overall": overall,
        "release_blocked": overall != "pass",
        "samples": metrics,
        "groups": groups,
        "parallax_fit": parallax_fit,
        "fit": {
            "tested": len(default_clearance),
            "fit": fit_count,
            "too_tight": too_tight,
            "rate": fit_rate,
        },
        "corrections": sum(sample.outcome.corrections or 0 for sample in measured),
        "recaptures": sum(sample.outcome.recaptures or 0 for sample in measured),
        "gates": gates,
    }


def report_markdown(report: dict) -> str:
    lines = [
        f"# G1 physical accuracy report: {report['run_id']}",
        "",
        f"Overall: **{report['overall'].upper()}**",
        "",
        "| Gate | State | Value | Limit |",
        "| --- | --- | ---: | ---: |",
    ]
    for name, gate in report["gates"].items():
        value = "—" if gate["value"] is None else str(gate["value"])
        limit = "—" if gate["limit"] is None else str(gate["limit"])
        lines.append(
            f"| {name.replace('_', ' ')} | {gate['state']} | {value} | {limit} |"
        )
    lines.extend(["", "## Groups", ""])
    for name, group in sorted(report["groups"].items()):
        boundary = group["boundary_error_mm"]
        boundary_text = (
            f"p95 {boundary['p95']:.3f} mm"
            if boundary is not None
            else "no boundary data"
        )
        thickness = group["thickness_error_mm"]
        thickness_text = (
            f"thickness p95 {thickness['p95']:.3f} mm"
            if thickness is not None
            else "no thickness data"
        )
        lines.append(
            f"- **{name}**: {group['samples']} samples, {boundary_text}, "
            f"{thickness_text}, "
            f"{group['corrections']} corrections, {group['recaptures']} recaptures"
        )
    if report["parallax_fit"] is not None:
        fit = report["parallax_fit"]
        lines.extend(
            [
                "",
                "## Residual parallax fit",
                "",
                (
                    f"error = {fit['intercept_mm']:.4f} + "
                    f"{fit['slope']:.4f} × (r·t/H), n={fit['samples']}"
                ),
            ]
        )
    lines.extend(
        [
            "",
            (
                "Release remains blocked."
                if report["release_blocked"]
                else "All physical release gates pass."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _svg_path(poly: Poly) -> str:
    paths = []
    for ring in [poly.exterior, *poly.holes]:
        points = " L ".join(f"{x:.3f} {y:.3f}" for x, y in ring)
        paths.append(f"M {points} Z")
    return " ".join(paths)


def _apply_registration(poly: Poly, reference: Poly, registration: dict) -> Poly:
    shape = contour_mod.to_shapely(poly)
    origin = contour_mod.to_shapely(reference).centroid
    shape = rotate_shape(
        shape,
        registration["rotation_deg"],
        origin=(origin.x, origin.y),
        use_radians=False,
    )
    shape = translate_shape(
        shape,
        xoff=registration["translation_x_mm"],
        yoff=registration["translation_y_mm"],
    )
    return contour_mod.from_shapely(shape)


def gauge_svg(sample: G1Sample) -> str:
    """Printable 1:1 gauge with a mandatory 20mm scale-verification square."""
    geometry = sample.geometry
    if geometry.corrected_outline is None:
        raise ValueError(f"sample {sample.id} has no corrected outline")
    corrected = geometry.corrected_outline
    compensated = geometry.compensated_outline
    truth = geometry.truth_outline
    if truth is not None:
        aligned, registration = register_outline(corrected, truth)
        if compensated is not None:
            compensated = _apply_registration(
                compensated,
                corrected,
                registration,
            )
        corrected = aligned

    layers: list[tuple[str, str, Poly]] = [
        ("corrected physical outline", "#155eef", corrected)
    ]
    if compensated is not None:
        layers.append(("compensated pocket", "#d65a00", compensated))
    if truth is not None:
        layers.append(("physical truth", "#111111", truth))
    points = [
        (x, y)
        for _, _, poly in layers
        for ring in [poly.exterior, *poly.holes]
        for x, y in ring
    ]
    xs, ys = zip(*points)
    pad = 8.0
    header = 34.0
    minx, maxx = min(xs) - pad, max(xs) + pad
    miny, maxy = min(ys) - header, max(ys) + pad
    width, height = maxx - minx, maxy - miny
    paths = "\n".join(
        (
            f'<path d="{_svg_path(poly)}" fill="none" stroke="{colour}" '
            f'stroke-width="0.18"><title>{html.escape(label)}</title></path>'
        )
        for label, colour, poly in layers
    )
    legend = "\n".join(
        (
            f'<text x="{minx + 24:.3f}" y="{miny + 5 + index * 4:.3f}" '
            f'font-size="3" fill="{colour}">{html.escape(label)}</text>'
        )
        for index, (label, colour, _) in enumerate(layers)
    )
    title = html.escape(f"{sample.id} — print at 100% / Actual Size")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.3f}mm" '
        f'height="{height:.3f}mm" viewBox="{minx:.3f} {miny:.3f} '
        f'{width:.3f} {height:.3f}">\n'
        f'<text x="{minx:.3f}" y="{miny + 3:.3f}" font-size="3.2">{title}</text>\n'
        f'<rect x="{minx:.3f}" y="{miny + 5:.3f}" width="20" height="20" '
        'fill="none" stroke="#111" stroke-width="0.18"/>\n'
        f'<text x="{minx:.3f}" y="{miny + 4.5:.3f}" font-size="2.4">'
        'verify this square is 20.00 × 20.00 mm</text>\n'
        f'{legend}\n{paths}\n</svg>\n'
    )


def write_gauge(sample: G1Sample, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(gauge_svg(sample))
    return path
