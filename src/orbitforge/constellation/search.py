"""Search usable Walker combinations against service metrics.

The design space is the Cartesian product of altitudes, inclinations, plane
counts, per-plane satellite counts and Walker phasings F. Each distinct
T/P/F pattern is:

1. geometrically pre-screened (latitude reach of ground track + footprint);
2. propagated over a coarse grid/window and verified criterion by criterion;
3. rejected patterns keep their explicit rejection reasons;
4. accepted patterns are ranked (fewest satellites, then largest margin).

A final fine-grid re-evaluation of the best candidate confirms the result at
production fidelity.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from orbitforge.constellation.design import ConstellationPattern
from orbitforge.constellation.evaluator import (
    GridCell,
    ServiceMetrics,
    area_weighted_grid,
    evaluate_pattern,
)
from orbitforge.constellation.requirements import (
    ServiceRequirements,
    Verdict,
    geometric_prescreen,
    verify,
)
from orbitforge.orbits.periods import orbital_period_s


@dataclass
class DesignSpace:
    altitudes_km: list[float]
    inclinations_deg: list[float]
    planes: list[int]
    per_plane: list[int]
    phasings: Optional[list[int]] = None  # None -> all valid F = 0..P_s-1

    def enumerate_patterns(self, name_prefix: str = "Walker") -> list[ConstellationPattern]:
        seen = set()
        out = []
        for alt in self.altitudes_km:
            for inc_deg in self.inclinations_deg:
                inc = math.radians(inc_deg)
                for planes in self.planes:
                    for spp in self.per_plane:
                        total = planes * spp
                        fs = range(spp) if self.phasings is None else self.phasings
                        for f in fs:
                            if not 0 <= f < spp:
                                continue
                            key = (alt, round(inc_deg, 6), planes, spp, f)
                            if key in seen:
                                continue
                            seen.add(key)
                            out.append(
                                ConstellationPattern(
                                    satellites=total,
                                    planes=planes,
                                    phasing=f,
                                    alt_km=float(alt),
                                    inclination_rad=inc,
                                    name=name_prefix,
                                )
                            )
        return out


@dataclass
class CandidateResult:
    pattern: ConstellationPattern
    accepted: bool
    verdict: Optional[Verdict] = None
    metrics: Optional[ServiceMetrics] = None
    rejection_stage: str = ""  # "geometric" or "simulation"
    rejection_reasons: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def pattern_id(self):
        return self.pattern.pattern_id


@dataclass
class SearchConfig:
    duration_s: float = 2.0 * 86400.0
    step_s: float = 300.0
    grid_lat_step_deg: float = 7.5
    grid_lon_step_deg: float = 7.5
    epoch_s: float = 0.0
    # Repeat the window at several constellation phases (fractions of an
    # orbital period) so the gap verdict does not depend on the start epoch.
    phase_offset_fractions: tuple[float, ...] = (0.25, 0.5, 0.75)


@dataclass
class SearchReport:
    requirements: ServiceRequirements
    config: SearchConfig
    candidates: list[CandidateResult] = field(default_factory=list)
    accepted: list[CandidateResult] = field(default_factory=list)
    rejected: list[CandidateResult] = field(default_factory=list)
    elapsed_s: float = 0.0

    def best(self) -> Optional[CandidateResult]:
        return self.accepted[0] if self.accepted else None


def _margin(verdict: Verdict) -> dict:
    """Worst normalised slack across coverage and gap criteria."""
    m = {}
    for c in verdict.criteria:
        if c.actual is None or c.required is None:
            continue
        if "coverage" in c.name:
            m[c.name] = c.actual - c.required
        elif "gap" in c.name and c.required > 0:
            # Positive slack when actual gap is well below the limit.
            m[c.name] = (c.required - c.actual) / c.required
    return m


def rank_key(cand: CandidateResult):
    margins = _margin(cand.verdict)
    worst = min(margins.values()) if margins else -1.0
    # Fewer satellites first; for equal size the worst-criterion margin decides,
    # then lower altitude (cheaper launch / shorter link range).
    return (cand.pattern.satellites, -worst, cand.pattern.alt_km)


def search_design_space(
    space: DesignSpace,
    requirements: ServiceRequirements,
    config: Optional[SearchConfig] = None,
    progress: Optional[Callable[[int, int, CandidateResult], None]] = None,
) -> SearchReport:
    config = config or SearchConfig()
    patterns = space.enumerate_patterns()
    report = SearchReport(requirements=requirements, config=config)
    grid = area_weighted_grid(config.grid_lat_step_deg, config.grid_lon_step_deg)
    t0 = time.perf_counter()

    for k, pattern in enumerate(patterns):
        cand = CandidateResult(pattern=pattern, accepted=False)
        tc = time.perf_counter()

        reason = geometric_prescreen(pattern, requirements)
        if reason is not None:
            cand.rejection_stage = "geometric"
            cand.rejection_reasons = [reason]
            cand.elapsed_s = time.perf_counter() - tc
            report.candidates.append(cand)
            report.rejected.append(cand)
            if progress:
                progress(k + 1, len(patterns), cand)
            continue

        period_s = orbital_period_s(pattern.semi_major_axis_km)
        offsets = [f * period_s for f in config.phase_offset_fractions]
        metrics = evaluate_pattern(
            pattern,
            duration_s=config.duration_s,
            step_s=config.step_s,
            min_elevation_rad=requirements.min_elevation_rad,
            revisit_target_s=requirements.max_global_gap_s,
            polar_latitude_deg=requirements.polar_latitude_deg,
            epoch_s=config.epoch_s,
            cells=grid,
            phase_offsets_s=offsets,
        )
        verdict = verify(metrics, requirements)
        cand.verdict = verdict
        cand.metrics = metrics
        cand.accepted = verdict.accepted
        cand.rejection_stage = "" if verdict.accepted else "simulation"
        cand.rejection_reasons = list(verdict.rejection_reasons)
        cand.elapsed_s = time.perf_counter() - tc
        report.candidates.append(cand)
        (report.accepted if verdict.accepted else report.rejected).append(cand)
        if progress:
            progress(k + 1, len(patterns), cand)

    report.accepted.sort(key=rank_key)
    report.elapsed_s = time.perf_counter() - t0
    return report


def refine_minimum_satellite_shortlist(
    report: SearchReport,
    requirements: ServiceRequirements,
    duration_s: float = 2.0 * 86400.0,
    step_s: float = 120.0,
    grid_lat_step_deg: float = 4.0,
    grid_lon_step_deg: float = 4.0,
    epoch_s: float = 0.0,
    max_candidates: int = 12,
):
    """Re-evaluate every accepted pattern at the minimum satellite count.

    Coarse-grid passes with identical satellite counts are tied; only fine-grid
    verification tells which phasings truly work. Candidates are refined from
    the smallest accepted T upward until at least one survives.

    Returns ``(survivors, tested)`` lists of :class:`CandidateResult`.
    """
    if not report.accepted:
        return [], []
    by_total = {}
    for cand in report.accepted:
        by_total.setdefault(cand.pattern.satellites, []).append(cand)

    survivors = []
    tested = []
    for total in sorted(by_total):
        group = by_total[total]

        def coarse_gap_key(c):
            gap = c.metrics.globe.worst_interior_gap_s
            return (gap if gap is not None else 1e18, c.pattern.alt_km, c.pattern.phasing)

        group = sorted(group, key=coarse_gap_key)[:max_candidates]
        for cand in group:
            fine = refine_candidate(
                cand,
                requirements,
                duration_s=duration_s,
                step_s=step_s,
                grid_lat_step_deg=grid_lat_step_deg,
                grid_lon_step_deg=grid_lon_step_deg,
                epoch_s=epoch_s,
            )
            tested.append(fine)
            if fine.accepted:
                survivors.append(fine)
        if survivors:
            survivors.sort(key=lambda c: (
                c.metrics.globe.worst_interior_gap_s or 0.0,
                c.pattern.alt_km,
                c.pattern.phasing,
            ))
            return survivors, tested
    return survivors, tested


def refine_candidate(
    cand: CandidateResult,
    requirements: ServiceRequirements,
    duration_s: float = 3.0 * 86400.0,
    step_s: float = 120.0,
    grid_lat_step_deg: float = 4.0,
    grid_lon_step_deg: float = 4.0,
    epoch_s: float = 0.0,
    phase_offset_fractions: tuple[float, ...] = (0.25, 0.5, 0.75),
) -> CandidateResult:
    """Re-evaluate one accepted candidate on a fine grid / long window."""
    offsets = [f * orbital_period_s(cand.pattern.semi_major_axis_km) for f in phase_offset_fractions]
    metrics = evaluate_pattern(
        cand.pattern,
        duration_s=duration_s,
        step_s=step_s,
        min_elevation_rad=requirements.min_elevation_rad,
        revisit_target_s=requirements.max_global_gap_s,
        polar_latitude_deg=requirements.polar_latitude_deg,
        epoch_s=epoch_s,
        grid_lat_step_deg=grid_lat_step_deg,
        grid_lon_step_deg=grid_lon_step_deg,
        phase_offsets_s=offsets,
    )
    verdict = verify(metrics, requirements)
    out = CandidateResult(
        pattern=cand.pattern,
        accepted=verdict.accepted,
        verdict=verdict,
        metrics=metrics,
        rejection_stage="" if verdict.accepted else "simulation",
        rejection_reasons=list(verdict.rejection_reasons),
    )
    return out
