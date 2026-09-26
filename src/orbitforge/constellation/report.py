"""Render constellation plans: per-satellite slot + orbit table and verdicts."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from typing import Optional

from orbitforge.constellation.design import (
    ConstellationPattern,
    Satellite,
    instantiate,
)
from orbitforge.constellation.evaluator import RegionMetrics, ServiceMetrics
from orbitforge.constellation.requirements import ServiceRequirements, Verdict
from orbitforge.constellation.search import CandidateResult, SearchReport

from .design import DEG


def satellite_rows(design):
    """One row per satellite with slot and full orbital parameters."""
    rows = []
    for s in design.satellites:
        rows.append(
            {
                "index": s.index,
                "name": s.name,
                "plane": s.plane,
                "slot": s.slot,
                "a_km": round(s.a_km, 3),
                "alt_km": round(s.alt_km, 3),
                "e": s.e,
                "i_deg": round(s.i_rad * DEG, 4),
                "raan_deg": round(s.raan_rad * DEG, 4),
                "argp_deg": round(s.argp_rad * DEG, 4),
                "mean_anomaly_deg": round(s.mean_anomaly_rad * DEG, 4),
                "period_min": round(s.period_s / 60.0, 3),
            }
        )
    return rows


def _region_lines(rm: Optional[RegionMetrics], label: str, indent="  "):
    if rm is None:
        return [f"{indent}{label}: no cells"]
    lines = [
        f"{indent}{label} ({rm.cell_count} cells, {rm.weight_fraction * 100:.1f}% of global area):",
        f"{indent}  ever-served coverage : {rm.ever_coverage * 100:7.2f} %",
        f"{indent}  continuous coverage  : {rm.continuous_coverage * 100:7.2f} %",
        f"{indent}  mean instantaneous   : {rm.mean_instantaneous_coverage * 100:7.2f} %",
    ]
    if rm.worst_interior_gap_s is not None:
        lines.append(f"{indent}  worst revisit gap    : {rm.worst_interior_gap_s / 3600:7.2f} h")
        lines.append(f"{indent}  p90 revisit gap      : {(rm.p90_interior_gap_s or 0) / 3600:7.2f} h")
        lines.append(f"{indent}  mean revisit gap     : {(rm.mean_interior_gap_s or 0) / 3600:7.2f} h")
        lines.append(f"{indent}  on-cadence fraction  : {rm.served_on_cadence * 100:7.2f} %")
        lines.append(f"{indent}  worst initial wait   : {(rm.worst_initial_wait_s or 0) / 3600:7.2f} h")
        lines.append(f"{indent}  worst tail wait      : {(rm.worst_tail_gap_s or 0) / 3600:7.2f} h")
    else:
        lines.append(f"{indent}  worst revisit gap    :     n/a (no revisits)")
    lines.append(f"{indent}  min fold (served)    : {rm.min_fold}")
    lines.append(f"{indent}  never-served cells   : {rm.unserved_cells}")
    return lines


def render_metrics(metrics: ServiceMetrics) -> str:
    lines = [
        f"Service metrics for {metrics.pattern_id}",
        f"  satellites {metrics.satellites} | window {metrics.duration_s / 86400:.2f} d "
        f"| step {metrics.step_s:.0f} s | grid {metrics.grid_lat_step_deg:g}x{metrics.grid_lon_step_deg:g} deg",
        f"  min elevation {metrics.min_elevation_rad * DEG:.1f} deg | "
        f"footprint half-angle {metrics.footprint_half_angle_rad * DEG:.2f} deg | "
        f"period {metrics.period_s / 60:.1f} min",
    ]
    lines += _region_lines(metrics.globe, "GLOBAL")
    lines += _region_lines(metrics.polar, f"POLAR CAPS >= {metrics.polar.region.split('_')[-1]}")
    lines += _region_lines(metrics.north_polar, "North polar", indent="    ")
    lines += _region_lines(metrics.south_polar, "South polar", indent="    ")
    return "\n".join(lines)


def render_verdict(verdict: Verdict) -> str:
    status = "ACCEPT" if verdict.accepted else "REJECT"
    lines = [f"Verdict for {verdict.pattern_id}: {status}"]
    for c in verdict.criteria:
        mark = "PASS" if c.passed else "FAIL"
        line = f"  [{mark}] {c.name}"
        if not c.passed and c.reason:
            line += f"  -> {c.reason.split(': ', 1)[-1]}"
        lines.append(line)
    for w in verdict.warnings:
        lines.append(f"  [warn] {w}")
    if not verdict.accepted:
        lines.append("  Rejection reasons:")
        for r in verdict.rejection_reasons:
            lines.append(f"    - {r}")
    return "\n".join(lines)


def render_plan(pattern: ConstellationPattern, metrics: Optional[ServiceMetrics],
                verdict: Optional[Verdict], epoch_s: float = 0.0) -> str:
    design = instantiate(pattern, epoch_s=epoch_s)
    rows = satellite_rows(design)
    lines = [
        "=" * 78,
        f"CONSTELLATION PLAN: {pattern.pattern_id}",
        "=" * 78,
        f"Walker delta T/P/F = {pattern.satellites}/{pattern.planes}/{pattern.phasing}; "
        f"{pattern.per_plane} satellites per plane",
        f"altitude {pattern.alt_km:g} km | inclination {pattern.inclination_rad * DEG:.2f} deg | "
        f"eccentricity {pattern.eccentricity} | period {design.period_s / 60:.2f} min",
        f"epoch (TAI s) {epoch_s}",
        "",
        f"{'#':>3} {'name':>8} {'plane':>5} {'slot':>4} {'a_km':>9} {'alt_km':>8} "
        f"{'i_deg':>7} {'RAAN_deg':>9} {'M0_deg':>8} {'period_min':>10}",
        "-" * 78,
    ]
    for r in rows:
        lines.append(
            f"{r['index']:>3} {r['name']:>8} {r['plane']:>5} {r['slot']:>4} "
            f"{r['a_km']:>9.3f} {r['alt_km']:>8.3f} {r['i_deg']:>7.3f} "
            f"{r['raan_deg']:>9.3f} {r['mean_anomaly_deg']:>8.3f} {r['period_min']:>10.3f}"
        )
    if metrics is not None:
        lines += ["", render_metrics(metrics)]
    if verdict is not None:
        lines += ["", render_verdict(verdict)]
    return "\n".join(lines)


def render_search_report(report: SearchReport, show_rejections: int = 12,
                         show_accepted: int = 10) -> str:
    lines = [
        "=" * 78,
        f"CONSTELLATION SEARCH: {len(report.candidates)} candidates, "
        f"{len(report.accepted)} accepted, {len(report.rejected)} rejected "
        f"({report.elapsed_s:.1f} s)",
        "=" * 78,
    ]
    lines.append("Accepted combinations (ranked):")
    if not report.accepted:
        lines.append("  (none)")
    for cand in report.accepted[:show_accepted]:
        g = cand.metrics.globe
        p = cand.metrics.polar
        lines.append(
            f"  {cand.pattern.pattern_id:<42} T={cand.pattern.satellites:<4} "
            f"glob {g.ever_coverage * 100:6.2f}% gap {g.worst_interior_gap_s / 3600:5.2f}h | "
            f"polar {p.ever_coverage * 100:6.2f}% gap {p.worst_interior_gap_s / 3600:5.2f}h"
        )
    geom = [c for c in report.rejected if c.rejection_stage == "geometric"]
    sim = [c for c in report.rejected if c.rejection_stage == "simulation"]
    lines.append("")
    lines.append(f"Rejected at geometry pre-screen: {len(geom)}")
    reason_groups = {}
    for c in geom:
        key = c.rejection_reasons[0].split(": ground")[0]
        reason_groups.setdefault(key, []).append(c.pattern.pattern_id)
    for key, ids in list(reason_groups.items())[:show_rejections]:
        lines.append(f"  - {key} ({len(ids)} patterns)")
        lines.append(f"      e.g. {', '.join(ids[:3])}")
    lines.append("")
    lines.append(f"Rejected after simulation ({len(sim)}), first {min(show_rejections, len(sim))}:")
    for cand in sim[:show_rejections]:
        lines.append(f"  x {cand.pattern.pattern_id}")
        for r in cand.rejection_reasons:
            lines.append(f"      - {r}")
    return "\n".join(lines)


def _dataclass_to_dict(obj):
    if is_dataclass(obj):
        return {k: _dataclass_to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def plan_to_dict(pattern: ConstellationPattern, metrics: Optional[ServiceMetrics],
                 verdict: Optional[Verdict], epoch_s: float = 0.0) -> dict:
    design = instantiate(pattern, epoch_s=epoch_s)
    out = {
        "pattern": {
            "id": pattern.pattern_id,
            "satellites": pattern.satellites,
            "planes": pattern.planes,
            "phasing": pattern.phasing,
            "per_plane": pattern.per_plane,
            "alt_km": pattern.alt_km,
            "inclination_deg": pattern.inclination_rad * DEG,
            "eccentricity": pattern.eccentricity,
            "argp_deg": pattern.argp_rad * DEG,
            "period_s": design.period_s,
            "epoch_s": epoch_s,
        },
        "satellites": satellite_rows(design),
    }
    if metrics is not None:
        out["metrics"] = _dataclass_to_dict(metrics)
    if verdict is not None:
        out["verdict"] = _dataclass_to_dict(verdict)
    return out


def report_to_dict(report: SearchReport) -> dict:
    def cand_dict(c: CandidateResult):
        return {
            "pattern": {
                "id": c.pattern.pattern_id,
                "T": c.pattern.satellites,
                "P": c.pattern.planes,
                "F": c.pattern.phasing,
                "alt_km": c.pattern.alt_km,
                "inclination_deg": c.pattern.inclination_rad * DEG,
            },
            "accepted": c.accepted,
            "stage": c.rejection_stage or "accepted",
            "reasons": c.rejection_reasons,
            "metrics": _dataclass_to_dict(c.metrics) if c.metrics else None,
        }

    return {
        "requirements": _dataclass_to_dict(report.requirements),
        "config": _dataclass_to_dict(report.config),
        "elapsed_s": report.elapsed_s,
        "accepted": [cand_dict(c) for c in report.accepted],
        "rejected": [cand_dict(c) for c in report.rejected],
    }


def write_json(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
