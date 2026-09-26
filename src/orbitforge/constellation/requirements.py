"""Service requirements and explicit pass/fail verdicts.

Every requirement is checked criterion by criterion. A failing combination is
rejected with one reason per failed metric (global coverage, global worst
revisit gap, polar coverage, polar cadence, continuous fold). A mean coverage
number alone can never produce a pass.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from orbitforge.constellation.design import (
    ConstellationPattern,
    access_half_angle_rad,
    max_ground_latitude_rad,
)
from orbitforge.constellation.evaluator import RegionMetrics, ServiceMetrics

from .design import DEG


@dataclass(frozen=True)
class ServiceRequirements:
    min_global_coverage: float = 1.0
    max_global_gap_s: Optional[float] = None
    min_polar_coverage: float = 1.0
    max_polar_gap_s: Optional[float] = None
    min_continuous_coverage: float = 0.0
    min_elevation_rad: float = 0.0
    polar_latitude_deg: float = 75.0
    minimum_simulation_days: float = 2.0

    @classmethod
    def degrees(cls, min_elevation_deg=0.0, polar_latitude_deg=75.0, **kw):
        return cls(
            min_elevation_rad=math.radians(min_elevation_deg),
            polar_latitude_deg=polar_latitude_deg,
            **kw,
        )


@dataclass
class CriterionResult:
    name: str
    passed: bool
    actual: Optional[float]
    required: Optional[float]
    reason: str = ""


@dataclass
class Verdict:
    pattern_id: str
    accepted: bool
    criteria: list[CriterionResult] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: Optional[ServiceMetrics] = None

    def criterion(self, name):
        for c in self.criteria:
            if c.name == name:
                return c
        return None


def _fmt_hours(seconds):
    if seconds is None:
        return "n/a"
    return f"{seconds / 3600.0:.2f} h"


def geometric_reach(pattern: ConstellationPattern, min_elevation_rad: float):
    """Maximum serviceable latitude from inclination + footprint geometry."""
    alpha = access_half_angle_rad(pattern.alt_km, min_elevation_rad)
    return max_ground_latitude_rad(pattern.inclination_rad) + alpha


def geometric_prescreen(pattern: ConstellationPattern, req: ServiceRequirements):
    """Cheap single-orbit reachability screen.

    Returns None if the pattern *might* satisfy ever-coverage, or a rejection
    reason. Revisit gaps and continuous coverage cannot be screened this way and
    always go to full simulation.
    """
    reach = geometric_reach(pattern, req.min_elevation_rad)
    polar_lat = math.radians(req.polar_latitude_deg)
    if req.min_polar_coverage > 0.0 and reach < polar_lat:
        return (
            f"geometric reach {reach * DEG:.1f} deg < polar latitude "
            f"{req.polar_latitude_deg:g} deg: ground track max latitude "
            f"{max_ground_latitude_rad(pattern.inclination_rad) * DEG:.1f} deg plus "
            f"footprint {access_half_angle_rad(pattern.alt_km, req.min_elevation_rad) * DEG:.1f} deg "
            "cannot see the polar caps"
        )
    if req.min_global_coverage >= 0.999 and reach < math.pi / 2.0:
        return (
            f"geometric reach {reach * DEG:.1f} deg < 90 deg: global coverage impossible "
            "at the required minimum elevation"
        )
    return None


def verify(metrics: ServiceMetrics, req: ServiceRequirements) -> Verdict:
    """Apply every requirement and build an explicit accept/reject verdict."""
    verdict = Verdict(pattern_id=metrics.pattern_id, accepted=True, metrics=metrics)

    def check_ratio(name, actual, required, better=">="):
        passed = actual is not None and (actual >= required if better == ">=" else actual <= required)
        at = f"{actual:.4f}" if actual is not None else "n/a"
        reason = "" if passed else f"{name}: {at} {better} required {required:.4f}"
        verdict.criteria.append(CriterionResult(name, passed, actual, required, reason))
        if not passed:
            verdict.accepted = False
            verdict.rejection_reasons.append(reason)

    def check_gap(name, region: RegionMetrics, required):
        gap = region.worst_interior_gap_s
        if region.unserved_cells:
            reason = (
                f"{name}: {region.unserved_cells} cells never served; no service gap "
                f"can satisfy <= {_fmt_hours(required)} for them"
            )
            passed = False
        elif gap is None:
            reason = (
                f"{name}: no cell revisited within the window "
                f"(required worst gap <= {_fmt_hours(required)})"
            )
            passed = False
        elif region.unresolved_cadence_cells:
            reason = (
                f"{name}: {region.unresolved_cadence_cells} cells "
                f"({region.unresolved_cadence_fraction * 100:.2f}% of area) had fewer than "
                "two access windows in the simulation: cadence cannot be proven "
                f"(required worst gap <= {_fmt_hours(required)})"
            )
            passed = False
        else:
            passed = gap <= required
            reason = (
                ""
                if passed
                else f"{name}: worst wait {_fmt_hours(gap)} > required {_fmt_hours(required)}"
            )
        verdict.criteria.append(CriterionResult(name, passed, gap, required, reason))
        if not passed:
            verdict.accepted = False
            verdict.rejection_reasons.append(reason)

    g: RegionMetrics = metrics.globe
    p: RegionMetrics = metrics.polar

    check_ratio("global_ever_coverage", g.ever_coverage, req.min_global_coverage)
    if req.max_global_gap_s is not None:
        check_gap("global_revisit_gap", g, req.max_global_gap_s)
    if req.min_polar_coverage > 0.0:
        check_ratio("polar_ever_coverage", p.ever_coverage, req.min_polar_coverage)
    if req.max_polar_gap_s is not None:
        check_gap("polar_revisit_gap", p, req.max_polar_gap_s)
    if req.min_continuous_coverage > 0.0:
        check_ratio("continuous_coverage", g.continuous_coverage, req.min_continuous_coverage)

    # Window adequacy warning: interior gaps are only meaningful if the window
    # comfortably exceeds the target cadence.
    if req.max_global_gap_s is not None:
        need = req.minimum_simulation_days * 86400.0
        if metrics.duration_s < need:
            verdict.warnings.append(
                f"simulation window {metrics.duration_s / 86400.0:.2f} d shorter than "
                f"recommended {req.minimum_simulation_days:.2f} d for gap statistics"
            )
    if g.unserved_cells:
        verdict.warnings.append(
            f"{g.unserved_cells} grid cells never visible during the window"
        )
    if g.worst_initial_wait_s is not None and req.max_global_gap_s is not None:
        edge = max(g.worst_initial_wait_s, g.worst_tail_gap_s or 0.0)
        # With continuous coverage every cell is visible at the window edges,
        # so the "tail wait" is not a real outage.
        if g.continuous_coverage < 1.0 and edge > req.max_global_gap_s:
            verdict.warnings.append(
                "edge-of-window wait "
                f"{_fmt_hours(edge)} is counted in the worst service gap; "
                "extend the window if this is a campaign boundary artefact"
            )

    return verdict
