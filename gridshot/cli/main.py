"""GridShot CLI: mat generation and verification (M0)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import typer

from gridshot.core import calibrate as calibrate_mod
from gridshot.core import devices as devices_mod
from gridshot.core import ingest as ingest_mod
from gridshot.core import mat as mat_mod

app = typer.Typer(no_args_is_help=True, help="Phone photo → Gridfinity tool-cutout bins.")
mat_app = typer.Typer(no_args_is_help=True, help="Calibration mat: generate, verify, list.")
calib_app = typer.Typer(no_args_is_help=True, help="Camera calibration: per-device intrinsics.")
bench_app = typer.Typer(no_args_is_help=True, help="Printer calibration and physical G1 accuracy.")
app.add_typer(mat_app, name="mat")
app.add_typer(calib_app, name="calib")
app.add_typer(bench_app, name="bench")


@bench_app.command("coupon")
def bench_coupon(
    out: Path = typer.Option(Path("out"), help="Output directory."),
    copies: int = typer.Option(
        3,
        min=1,
        help="Independent copies to print and measure for uncertainty.",
    ),
) -> None:
    """Export the long-baseline coupon used for repeated measurements."""
    from gridshot.core import bench as bench_mod
    from gridshot.core import export as export_mod
    from gridshot.core import gridfinity as grid_mod2

    mesh = grid_mod2.to_trimesh(bench_mod.coupon_solid())
    files = export_mod.write_all(out, "gridshot-coupon", mesh)
    for kind, output_path in files.items():
        typer.echo(f"{kind}:  {output_path}")
    typer.echo("")
    typer.echo(
        f"Print {copies} independent copies with the same printer/material/nozzle/process."
    )
    typer.echo("Measure both pockets with inside jaws at mid-depth, repeating each option:")
    typer.echo(
        "  gridshot bench record --printer NAME --material MATERIAL "
        "--nozzle-mm 0.4 --process PROFILE"
    )
    typer.echo("    --a-x <copy1> --a-x <copy2> --a-x <copy3> ...")
    typer.echo(
        f"  nominals: A {bench_mod.A_X}x{bench_mod.A_Y}, "
        f"B {bench_mod.B_X}x{bench_mod.B_Y}"
    )


@bench_app.command("record")
def bench_record(
    a_x: list[float] = typer.Option(..., help="Long-pocket X; repeat per copy."),
    a_y: list[float] = typer.Option(..., help="Long-pocket Y; repeat per copy."),
    b_x: list[float] = typer.Option(..., help="Short-pocket X; repeat per copy."),
    b_y: list[float] = typer.Option(..., help="Short-pocket Y; repeat per copy."),
    printer: str = typer.Option("unspecified", help="Printer identifier."),
    material: str = typer.Option("unspecified", help="Material and formulation."),
    nozzle_mm: float = typer.Option(0.4, min=0.1, help="Nozzle diameter."),
    process: str = typer.Option("unspecified", help="Slicer/process profile."),
) -> None:
    """Fit repeated coupon measurements and store an immutable profile revision."""
    from gridshot.core import bench as bench_mod
    from gridshot.core.models import PrinterSignature

    counts = {len(a_x), len(a_y), len(b_x), len(b_y)}
    if len(counts) != 1:
        raise typer.BadParameter("a-x, a-y, b-x, and b-y need equal repeat counts")
    observations = [
        {"a_x": ax, "a_y": ay, "b_x": bx, "b_y": by}
        for ax, ay, bx, by in zip(a_x, a_y, b_x, b_y, strict=True)
    ]
    try:
        profile, warnings = bench_mod.solve_repeated(
            observations,
            signature=PrinterSignature(
                printer=printer,
                material=material,
                nozzle_mm=nozzle_mm,
                process=process,
            ),
        )
    except bench_mod.CouponMeasurementError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"profile: {profile.profile_id} revision {profile.revision}")
    typer.echo(f"quality: {profile.quality}")
    typer.echo(f"scale:   X {100 * profile.scale_x:+.2f}%   Y {100 * profile.scale_y:+.2f}%")
    typer.echo(f"offset:  {profile.offset_mm:.3f} mm per side")
    typer.echo(
        "repeat:  max range "
        f"{profile.uncertainty.get('max_repeat_range_mm', 0.0):.3f} mm"
    )
    for warning in warnings:
        typer.secho(f"warning: {warning}", fg=typer.colors.YELLOW)
    if profile.quality == "review":
        typer.secho(
            "profile retained for diagnosis but not activated",
            fg=typer.colors.YELLOW,
        )
    else:
        typer.secho(
            "printer profile activated — future pockets use this exact revision",
            fg=typer.colors.GREEN,
        )


@bench_app.command("printer-profiles")
def bench_printer_profiles() -> None:
    """List immutable printer compensation revisions."""
    from gridshot.core import bench as bench_mod

    active = bench_mod.load_profile()
    for profile in bench_mod.list_profiles():
        marker = "*" if active and (
            active.profile_id, active.revision
        ) == (profile.profile_id, profile.revision) else " "
        signature = profile.signature
        typer.echo(
            f"{marker} {profile.profile_id} v{profile.revision} [{profile.quality}] "
            f"{signature.printer} / {signature.material} / "
            f"{signature.nozzle_mm:g}mm / {signature.process}"
        )


@bench_app.command("printer-activate")
def bench_printer_activate(
    profile_id: str = typer.Argument(help="Printer profile id."),
    revision: Optional[int] = typer.Option(None, min=1, help="Revision; newest by default."),
) -> None:
    """Activate one passing immutable printer-profile revision."""
    from gridshot.core import bench as bench_mod

    try:
        profile = bench_mod.activate_profile(profile_id, revision)
    except (KeyError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"active: {profile.profile_id} revision {profile.revision}")



@bench_app.command("g1-init")
def bench_g1_init(
    run_dir: Path = typer.Argument(help="Directory for the physical G1 run."),
    run_id: Optional[str] = typer.Option(None, help="Run id; defaults to directory name."),
    operator: str = typer.Option("", help="Person performing the measurements."),
    printer: str = typer.Option("", help="Printer used for fit trials."),
    material: str = typer.Option("", help="Material used for fit trials."),
    nozzle_mm: Optional[float] = typer.Option(None, help="Nozzle diameter."),
    slicer_profile: str = typer.Option("", help="Slicer/profile identifier."),
) -> None:
    """Create the required ten-condition physical accuracy matrix."""
    from gridshot.core import g1 as g1_mod

    manifest = g1_mod.new_manifest(
        run_id or run_dir.name,
        environment=g1_mod.G1Environment(
            operator=operator,
            printer=printer,
            material=material,
            nozzle_mm=nozzle_mm,
            slicer_profile=slicer_profile,
        ),
    )
    path = g1_mod.save_manifest(manifest, run_dir / "manifest.json")
    (run_dir / "gauges").mkdir(parents=True, exist_ok=True)
    typer.echo(f"manifest: {path}")
    typer.echo(f"matrix:   {len(manifest.samples)} required capture conditions")
    typer.echo("next: capture a result, then run gridshot bench g1-ingest")


@bench_app.command("g1-ingest")
def bench_g1_ingest(
    manifest_path: Path = typer.Argument(help="G1 manifest.json."),
    sample_id: str = typer.Argument(help="Matrix sample id, such as flat-center."),
    result_json: Path = typer.Argument(help="Trace result.json to preserve."),
    tool_id: str = typer.Option(..., help="Stable physical tool identifier."),
) -> None:
    """Attach a trace result and its raw/corrected/compensated geometry."""
    import json
    import shutil

    from gridshot.core import g1 as g1_mod

    manifest = g1_mod.load_manifest(manifest_path)
    try:
        result = json.loads(result_json.read_text())
        sample = g1_mod.ingest_result(
            manifest,
            sample_id,
            result,
            tool_id=tool_id,
        )
    except (KeyError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    run_dir = manifest_path.parent
    source_dir = run_dir / "sources" / sample_id
    source_names = list((result.get("provenance") or {}).get("source_images") or [])
    source_paths = [result_json.parent / Path(name).name for name in source_names]
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise typer.BadParameter(
            "source images referenced by result.json are missing: "
            + ", ".join(missing)
        )
    source_dir.mkdir(parents=True, exist_ok=True)
    result_snapshot = source_dir / "result.json"
    if result_json.resolve() != result_snapshot.resolve():
        shutil.copy2(result_json, result_snapshot)
    image_snapshots: list[str] = []
    for source_path in source_paths:
        snapshot = source_dir / source_path.name
        if source_path.resolve() != snapshot.resolve():
            shutil.copy2(source_path, snapshot)
        image_snapshots.append(str(snapshot.relative_to(run_dir)))
    sample.provenance.source_result = str(result_snapshot.relative_to(run_dir))
    sample.provenance.source_images = image_snapshots
    g1_mod.save_manifest(manifest, manifest_path)
    typer.echo(f"sample:   {sample.id} [{sample.status}]")
    typer.echo(f"manifest: {manifest_path}")
    typer.echo("next: print its gauge or record physical truth")


@bench_app.command("g1-record")
def bench_g1_record(
    manifest_path: Path = typer.Argument(help="G1 manifest.json."),
    sample_id: str = typer.Argument(help="Matrix sample id."),
    truth: Optional[Path] = typer.Option(
        None,
        help="Registered scanner/template truth polygon JSON.",
    ),
    truth_thickness: Optional[float] = typer.Option(
        None,
        help="Caliper truth thickness in millimetres.",
    ),
    fit: Optional[str] = typer.Option(
        None,
        help="untested, fit, too_tight, or too_loose.",
    ),
    corrections: Optional[int] = typer.Option(
        None,
        min=0,
        help="Manual outline corrections required.",
    ),
    recaptures: Optional[int] = typer.Option(
        None,
        min=0,
        help="Additional photo attempts required.",
    ),
    notes: Optional[str] = typer.Option(None, help="Physical measurement notes."),
) -> None:
    """Record physical outline/thickness truth and first-print outcome."""
    import shutil

    from gridshot.core import g1 as g1_mod

    manifest = g1_mod.load_manifest(manifest_path)
    truth_outline = g1_mod.load_poly(truth) if truth is not None else None
    try:
        truth_source = None
        if truth is not None:
            truth_dir = manifest_path.parent / "truth"
            truth_dir.mkdir(parents=True, exist_ok=True)
            truth_snapshot = truth_dir / f"{sample_id}.json"
            if truth.resolve() != truth_snapshot.resolve():
                shutil.copy2(truth, truth_snapshot)
            truth_source = str(
                truth_snapshot.relative_to(manifest_path.parent)
            )
        sample = g1_mod.record_sample(
            manifest,
            sample_id,
            truth_outline=truth_outline,
            truth_source=truth_source,
            truth_thickness_mm=truth_thickness,
            fit=fit,
            corrections=corrections,
            recaptures=recaptures,
            notes=notes,
        )
    except (KeyError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    g1_mod.save_manifest(manifest, manifest_path)
    typer.echo(f"sample:   {sample.id} [{sample.status}]")
    typer.echo(f"fit:      {sample.outcome.fit}")
    typer.echo(f"manifest: {manifest_path}")


@bench_app.command("g1-gauge")
def bench_g1_gauge(
    manifest_path: Path = typer.Argument(help="G1 manifest.json."),
    sample_id: Optional[str] = typer.Option(
        None,
        help="One sample id; omit to write every captured sample.",
    ),
    out: Optional[Path] = typer.Option(
        None,
        help="Output directory; defaults to a gauges folder beside the manifest.",
    ),
) -> None:
    """Write true-scale SVG outline gauges with a 20mm print check."""
    from gridshot.core import g1 as g1_mod

    manifest = g1_mod.load_manifest(manifest_path)
    if sample_id is not None:
        try:
            samples = [g1_mod.sample_by_id(manifest, sample_id)]
        except KeyError as exc:
            raise typer.BadParameter(f"unknown sample: {sample_id}") from exc
    else:
        samples = [
            sample
            for sample in manifest.samples
            if sample.geometry.corrected_outline is not None
        ]
    if not samples:
        raise typer.BadParameter("no captured samples have corrected outlines")
    out_dir = out or manifest_path.parent / "gauges"
    for sample in samples:
        try:
            path = g1_mod.write_gauge(
                sample,
                out_dir / f"{sample.id}.svg",
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        typer.echo(f"gauge: {path}")


@bench_app.command("g1-report")
def bench_g1_report(
    manifest_path: Path = typer.Argument(help="G1 manifest.json."),
    out: Optional[Path] = typer.Option(
        None,
        help="JSON report path; defaults beside the manifest.",
    ),
    enforce: bool = typer.Option(
        False,
        help="Exit nonzero unless every physical release gate passes.",
    ),
) -> None:
    """Calculate dimensional, fit, correction, and recapture release gates."""
    import json

    from gridshot.core import g1 as g1_mod

    manifest = g1_mod.load_manifest(manifest_path)
    report = g1_mod.build_report(manifest)
    report_path = out or manifest_path.parent / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    markdown_path = report_path.with_suffix(".md")
    markdown_path.write_text(g1_mod.report_markdown(report))
    typer.echo(f"overall:  {report['overall'].upper()}")
    for name, gate in report["gates"].items():
        typer.echo(f"  {name}: {gate['state']} — {gate['message']}")
    typer.echo(f"json:     {report_path}")
    typer.echo(f"markdown: {markdown_path}")
    if enforce and report["release_blocked"]:
        raise typer.Exit(code=1)


@mat_app.command("new")
def mat_new(
    paper: str = typer.Option("a4", help="Paper size: a4, a3, or letter."),
    out: Path = typer.Option(Path("mats"), help="Output directory for the PDF."),
) -> None:
    """Generate a printable ChArUco calibration mat PDF and register its profile."""
    spec = mat_mod.default_spec(paper)
    profile, pdf_path = mat_mod.new_mat(spec, out)
    typer.echo(f"mat id:  {profile.mat_id}")
    typer.echo(f"board:   {spec.squares_x}x{spec.squares_y} squares of {spec.square_mm} mm")
    typer.echo(f"pdf:     {pdf_path}")
    typer.echo("")
    typer.echo("Print at 100% / Actual Size on matte paper, tape it flat, then caliper")
    typer.echo(f"both black bars (expected {spec.span_mm:.2f} mm) and record them with:")
    typer.echo(f"  gridshot mat verify {profile.mat_id} --measured-x <mm> --measured-y <mm>")


@mat_app.command("verify")
def mat_verify(
    mat_id: str = typer.Argument(help="Mat id (see `gridshot mat list`)."),
    measured_x: Optional[float] = typer.Option(None, help="Calipered X bar length in mm."),
    measured_y: Optional[float] = typer.Option(None, help="Calipered Y bar length in mm."),
    photo: Optional[Path] = typer.Option(None, help="Check a phone photo of the printed mat."),
    force: bool = typer.Option(False, help="Accept a badly scaled print anyway."),
) -> None:
    """Record caliper measurements of the printed scale bars, or sanity-check a photo."""
    profile = mat_mod.load_profile(mat_id)

    if measured_x is not None or measured_y is not None:
        if measured_x is None or measured_y is None:
            raise typer.BadParameter("provide both --measured-x and --measured-y")
        profile, warnings = mat_mod.verify_scale(profile, measured_x, measured_y, force=force)
        typer.echo(f"scale:   X {profile.scale_x:.5f}  Y {profile.scale_y:.5f}")
        for w in warnings:
            typer.secho(f"warning: {w}", fg=typer.colors.YELLOW)
        if profile.verified:
            typer.secho("mat verified — it can now be used for calibration", fg=typer.colors.GREEN)
        else:
            raise typer.Exit(code=1)

    if photo is not None:
        src = ingest_mod.load(photo)
        prepared = devices_mod.prepare_image(src)
        device = prepared.profile
        if device is not None:
            typer.echo(f"device:  {device.device_id} (distortion-corrected)")
        cal = calibrate_mod.calibrate_image(
            prepared.pixels,
            profile,
            K=prepared.K,
            dist=None,
            exif=src.exif,
            allow_unverified=True,
            device_profile_id=device.device_id if device else None,
            device_profile_revision=device.revision if device else None,
            capture_signature=prepared.signature,
            intrinsics_source="profile" if device else None,
        )
        typer.echo(f"corners: {cal.n_corners}")
        typer.echo(f"rms:     {cal.reproj_rms_px:.2f} px")
        if cal.tilt_deg is not None:
            typer.echo(f"tilt:    {cal.tilt_deg:.1f} deg")
        if cal.camera_height_mm is not None:
            typer.echo(f"height:  {cal.camera_height_mm:.0f} mm")
        for w in cal.warnings:
            typer.secho(f"warning: {w}", fg=typer.colors.YELLOW)
        if not cal.warnings:
            typer.secho("photo check passed", fg=typer.colors.GREEN)

    if measured_x is None and photo is None:
        raise typer.BadParameter("nothing to do: pass --measured-x/--measured-y and/or --photo")


@mat_app.command("reference")
def mat_reference(
    mat_id: str = typer.Argument(help="Verified mat id."),
    photo: Path = typer.Argument(help="Photo of the EMPTY mat, exactly as taped down."),
) -> None:
    """Store an empty-mat reference photo — tool photos are diffed against it
    to locate tools for SAM segmentation."""
    import numpy as np

    from gridshot.core import diffseg as diffseg_mod

    profile = mat_mod.load_profile(mat_id)
    src = ingest_mod.load(photo)
    prepared = devices_mod.prepare_image(src)
    device = prepared.profile
    if device is None:
        typer.secho(
            "warning: no device profile — reference will carry lens distortion",
            fg=typer.colors.YELLOW,
        )
    cal = calibrate_mod.calibrate_image(
        prepared.pixels,
        profile,
        K=prepared.K,
        dist=None,
        exif=src.exif,
        device_profile_id=device.device_id if device else None,
        device_profile_revision=device.revision if device else None,
        capture_signature=prepared.signature,
        intrinsics_source="profile" if device else None,
    )
    pixels = prepared.pixels
    for w in cal.warnings:
        typer.secho(f"warning: {w}", fg=typer.colors.YELLOW)
    canonical = diffseg_mod.canonical_warp(pixels, cal, profile.spec)
    path = mat_mod.save_reference(mat_id, canonical)
    typer.echo(f"reference: {path} ({cal.n_corners} corners, rms {cal.reproj_rms_px:.2f}px)")
    typer.secho(
        "empty-mat reference stored — retake it whenever the mat or tape moves",
        fg=typer.colors.GREEN,
    )


@calib_app.command("intrinsics")
def calib_intrinsics(
    mat_id: str = typer.Argument(help="Verified mat the photos show."),
    photos_dir: Path = typer.Argument(help="Directory of 12+ photos of the mat from varied angles."),
    name: Optional[str] = typer.Option(None, help="Device profile name (default: from EXIF lens)."),
) -> None:
    """Calibrate a phone lens from photos of the mat; stores a device profile."""
    profile = mat_mod.load_profile(mat_id)
    if not profile.verified:
        typer.secho(f"mat '{mat_id}' is unverified — verify its print scale first", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    paths = sorted(
        p for p in photos_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic", ".webp"}
    )
    if not paths:
        raise typer.BadParameter(f"no photos found in {photos_dir}")
    typer.echo(f"loading {len(paths)} photos...")
    sources = [ingest_mod.load(p) for p in paths]

    try:
        signature = devices_mod.calibration_signature(sources)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    K, dist, rms, n_views, warnings = calibrate_mod.calibrate_intrinsics(
        [s.pixels for s in sources],
        profile,
        view_names=[path.name for path in paths],
    )
    for w in warnings:
        typer.secho(f"warning: {w}", fg=typer.colors.YELLOW)

    revision = devices_mod.next_revision(signature)
    device_profile = devices_mod.DeviceProfile(
        device_id=devices_mod.profile_id(signature, revision, name),
        revision=revision,
        created_at=datetime.now(timezone.utc).isoformat(),
        device_make=signature.device_make,
        device_model=signature.device_model,
        lens_model=signature.lens_model,
        image_size=signature.image_size,
        orientation_deg=signature.orientation_deg,
        focal_mm=signature.focal_mm,
        focal_35mm=signature.focal_35mm,
        digital_zoom_ratio=signature.digital_zoom_ratio,
        mat_id=profile.mat_id,
        n_views=n_views,
        source_images=[path.name for path in paths],
        K=K.tolist(),
        dist=dist.tolist(),
        reproj_rms_px=rms,
    )
    device = devices_mod.save_profile(device_profile)
    typer.echo(f"views:   {n_views} used of {len(paths)}")
    typer.echo(f"rms:     {rms:.3f} px")
    typer.echo(f"f:       {K[0][0]:.1f} px")
    typer.echo(f"profile: {device}")
    typer.secho(
        "device calibrated — photo checks and traces from this lens/resolution "
        "are now distortion-corrected automatically",
        fg=typer.colors.GREEN,
    )


@app.command("trace")
def trace(
    photo: Path = typer.Argument(help="Photo of one tool lying on the verified mat."),
    photo2: Optional[Path] = typer.Argument(
        None, help="Second photo from a different camera position — enables automatic thickness."
    ),
    thickness: Optional[float] = typer.Option(
        None, help="Height of the tool's widest outline, mm (omit when passing two photos)."
    ),
    clearance: float = typer.Option(
        1.0, help="Fit clearance around the outline, mm (0.5 tight / 1.0 snug / 1.5 loose)."
    ),
    smooth: float = typer.Option(0.6, help="Boundary noise suppression radius, mm (0 = off)."),
    style: Literal["pocket", "corral", "grid"] = typer.Option(
        "pocket", help="Bin style: recessed pocket, stackable corral, or live socket grid."
    ),
    depth: Optional[float] = typer.Option(
        None, help="Desired recess depth below the stacking plane, mm."
    ),
    full_height: Optional[float] = typer.Option(
        None, help="Measured maximum tool height, mm; drives automatic recess depth."
    ),
    height_u: Optional[int] = typer.Option(
        None, help="Bin height in 7mm gridfinity units (default: fewest that fit the pocket)."
    ),
    lip: bool = typer.Option(True, help="Spec stacking lip on the rim (--no-lip to omit)."),
    finger_hole: bool = typer.Option(False, help="Cut a 20mm finger hole in the pocket."),
    mat: Optional[str] = typer.Option(None, help="Mat id (default: the single verified mat)."),
    out: Path = typer.Option(Path("out"), help="Output directory."),
) -> None:
    """Trace a tool photo into a printable gridfinity bin (STL + 3MF + debug SVG)."""
    from gridshot.core import trace as trace_mod

    result = trace_mod.run(
        photo,
        thickness_mm=thickness,
        photo2=photo2,
        clearance_mm=clearance,
        smooth_mm=smooth,
        bin_style=style,
        pocket_depth_mm=depth,
        full_height_mm=full_height,
        height_u=height_u,
        lip=lip,
        finger_hole=finger_hole,
        mat_id=mat,
        out_dir=out,
    )
    cal = result.calibration
    typer.echo(f"calibration: {cal.n_corners} corners, rms {cal.reproj_rms_px:.2f}px, "
               f"height {cal.camera_height_mm:.0f}mm, tilt {cal.tilt_deg:.1f}°")
    k = 1.0 - result.thickness_mm / cal.camera_height_mm
    typer.echo(
        f"parallax:    shrink ×{k:.4f} toward nadir "
        f"(t={result.thickness_mm:.1f}mm, H={cal.camera_height_mm:.0f}mm)"
    )
    geometry = f"recess {result.pocket_depth_mm:.1f}mm deep"
    typer.echo(
        f"bin:         {result.grid[0]}x{result.grid[1]} units, "
        f"{result.height_u}u tall, {result.bin_style}, {geometry}"
    )
    for w in result.warnings:
        typer.secho(f"warning: {w}", fg=typer.colors.YELLOW)
    for kind, path in result.files.items():
        typer.echo(f"{kind}:  {path}")
    typer.secho("trace complete — slice the 3MF/STL in Bambu Studio", fg=typer.colors.GREEN)


@mat_app.command("list")
def mat_list() -> None:
    """List registered mat profiles."""
    profiles = mat_mod.list_profiles()
    if not profiles:
        typer.echo("no mats registered — run `gridshot mat new`")
        return
    for p in profiles:
        status = "verified" if p.verified else "UNVERIFIED"
        typer.echo(
            f"{p.mat_id}  {p.spec.paper}  {p.spec.squares_x}x{p.spec.squares_y}"
            f"  scale X {p.scale_x:.5f} Y {p.scale_y:.5f}  [{status}]"
        )


if __name__ == "__main__":
    app()
