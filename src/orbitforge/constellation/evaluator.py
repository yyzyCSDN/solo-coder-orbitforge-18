"""Service-level evaluation: global coverage, revisit gaps and polar service.

The evaluator instantiates a pattern, propagates every slot to subpoints over
a finite time window, and records for every cell of an area-weighted grid
whether at least one satellite is visible (above the minimum elevation).
Per-cell access windows yield the *worst interior revisit gap* -- the metric
that actually says whether a cell is served on cadence rather than "ever seen".

Aggregates are produced for the whole globe and for the polar caps separately,
both ever-served and instantaneous (continuous-fold) coverage are reported, and
:func:`evaluate_requirements` rejects combinations criterion by criterion with
explicit reasons instead of returning a single mean coverage number.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from orbitforge.constellation.design import (
    ConstellationPattern,
    access_half_angle_rad,
    instantiate,
)
from orbitforge.constellation.propagation import ConstellationSampler

TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class GridCell:
    index: int
    lat_rad: float
    lon_rad: float
    weight: float  # area weight (cos lat), sums to 1 over the globe


def area_weighted_grid(lat_step_deg: float = 5.0, lon_step_deg: float = 5.0):
    """Equal-angle grid with cos(latitude) area weights, normalised to 1."""
    cells = []
    lat = -90.0 + lat_step_deg / 2.0
    while lat < 90.0:
        lat_r = math.radians(lat)
        lon = -180.0 + lon_step_deg / 2.0
        row = []
        while lon < 180.0:
            row.append((lat_r, math.radians(lon)))
            lon += lon_step_deg
        w = math.cos(lat_r)
        for lat_rr, lon_rr in row:
            cells.append([lat_rr, lon_rr, w])
        lat += lat_step_deg
    total = sum(c[2] for c in cells)
    return [GridCell(i, c[0], c[1], c[2] / total) for i, c in enumerate(cells)]


@dataclass
class CellResult:
    index: int
    lat_rad: float
    lon_rad: float
    weight: float
    accesses: int = 0
    ever_visible: bool = False
    continuous: bool = True          # visible at every sampled step
    first_access_s: Optional[float] = None
    last_access_s: Optional[float] = None
    max_interior_gap_s: Optional[float] = None  # gap between access windows
    mean_gap_s: Optional[float] = None
    p90_gap_s: Optional[float] = None
    min_fold: int = 0
    max_fold: int = 0
    visible_steps: int = 0
    # Effective worst service gap: 0 if continuously visible, the largest gap
    # between access windows if revisited, None if seen once but not continuous
    # (cadence unprovable from the window).
    effective_worst_gap_s: Optional[float] = None
    worst_outage_s: Optional[float] = None


@dataclass
class RegionMetrics:
    region: str
    cell_count: int
    weight_fraction: float
    ever_coverage: float            # area-weighted fraction ever served
    continuous_coverage: float      # area-weighted fraction visible at EVERY sample
    mean_instantaneous_coverage: float
    worst_interior_gap_s: Optional[float]
    p90_interior_gap_s: Optional[float]
    mean_interior_gap_s: Optional[float]
    worst_initial_wait_s: Optional[float]
    worst_tail_gap_s: Optional[float]
    served_on_cadence: float        # area-weighted fraction with gaps <= target
    min_fold: Optional[int]
    gap_revisit_target_s: Optional[float] = None
    unserved_cells: int = 0
    unresolved_cadence_cells: int = 0
    unresolved_cadence_fraction: float = 0.0
    worst_gap_interior_only_s: Optional[float] = None


@dataclass
class ServiceMetrics:
    pattern_id: str
    satellites: int
    duration_s: float
    step_s: float
    grid_lat_step_deg: float
    grid_lon_step_deg: float
    min_elevation_rad: float
    footprint_half_angle_rad: float
    period_s: float
    cells: list[CellResult] = field(default_factory=list)
    globe: Optional[RegionMetrics] = None
    north_polar: Optional[RegionMetrics] = None
    south_polar: Optional[RegionMetrics] = None
    polar: Optional[RegionMetrics] = None

    def cell_at(self, lat_rad: float, lon_rad: float) -> Optional[CellResult]:
        best = None
        bd = 1e9
        for c in self.cells:
            dlon = abs((c.lon_rad - lon_rad + math.pi) % TWO_PI - math.pi)
            d = (c.lat_rad - lat_rad) ** 2 + dlon**2
            if d < bd:
                bd, best = d, c
        return best


def _simulate_cells(sampler: ConstellationSampler, cells, cos_alpha, sin_alpha,
                    duration_s, step_s):
    """Run the time loop and return per-cell access/gap statistics."""
    n_steps = max(1, int(round(duration_s / step_s))) + 1
    # Precomputed grid geometry.
    glat = [c.lat_rad for c in cells]
    glon = [c.lon_rad for c in cells]
    gsin = [math.sin(p) for p in glat]
    gcos = [math.cos(p) for p in glat]
    gsinlon = [math.sin(p) for p in glon]
    gcoslon = [math.cos(p) for p in glon]
    nc = len(cells)

    results = [
        CellResult(i, c.lat_rad, c.lon_rad, c.weight) for i, c in enumerate(cells)
    ]
    continuous = bytearray(b"\x01") * nc
    # Gap tracking via access window edges.
    prev_visible = bytearray(nc)
    access_times = [[] for _ in range(nc)]  # one representative time per window
    min_fold = [0] * nc
    max_fold = [0] * nc
    visible_steps = [0] * nc

    sin_a, cos_a = sin_alpha, cos_alpha
    band = math.asin(min(1.0, sin_a))  # |lat difference| beyond which no access

    for k in range(n_steps):
        t = k * step_s
        subpoints = sampler.sample(t)
        fold = [0] * nc
        for slat, slon in subpoints:
            ssin = math.sin(slat)
            scos = math.cos(slat)
            scos_sinlon = scos * math.sin(slon)
            scos_coslon = scos * math.cos(slon)
            lat_lo, lat_hi = slat - band, slat + band
            for j in range(nc):
                lat = glat[j]
                if lat < lat_lo or lat > lat_hi:
                    continue
                # cos(central angle) via dot product of unit position vectors.
                dot = ssin * gsin[j] + scos_sinlon * gsinlon[j] + scos_coslon * gcoslon[j]
                if dot >= cos_a:
                    fold[j] += 1
        for j in range(nc):
            f = fold[j]
            visible = f > 0
            if visible:
                visible_steps[j] += 1
                if not prev_visible[j]:
                    access_times[j].append(t)
                if f > max_fold[j]:
                    max_fold[j] = f
                if min_fold[j] == 0:
                    min_fold[j] = f
                elif f < min_fold[j]:
                    min_fold[j] = f
            else:
                continuous[j] = 0
            prev_visible[j] = 1 if visible else 0

    for j, r in enumerate(results):
        times = access_times[j]
        r.accesses = len(times)
        r.ever_visible = bool(times)
        r.continuous = bool(continuous[j])
        r.max_fold = max_fold[j]
        r.min_fold = min_fold[j] if times else 0
        r.visible_steps = visible_steps[j]
        if times:
            r.first_access_s = times[0]
            r.last_access_s = times[-1]
            gaps = [b - a for a, b in zip(times, times[1:])]
            if gaps:
                ordered = sorted(gaps)
                r.max_interior_gap_s = ordered[-1]
                r.mean_gap_s = sum(gaps) / len(gaps)
                idx = int(0.9 * (len(ordered) - 1))
                r.p90_gap_s = ordered[idx]
            if r.continuous:
                # Visible at every sampled step: no outage at all.
                r.effective_worst_gap_s = 0.0
                r.worst_outage_s = 0.0
            elif gaps:
                r.effective_worst_gap_s = r.max_interior_gap_s
                # Observed outages include the edge waits, which are real waits
                # experienced at the start/end of the campaign.
                r.worst_outage_s = max(
                    times[0], *(gaps), duration_s - times[-1]
                )
            else:
                # One short window: the unobserved stretch around it bounds the
                # outage conservatively from above.
                r.worst_outage_s = max(times[0], duration_s - times[-1])
            # One access window that does not span the whole run leaves
            # effective_worst_gap_s None: cadence cannot be demonstrated.
    return results, n_steps


def _weighted_quantile(pairs, q):
    """Quantile of (value, weight) pairs by cumulative weight."""
    pairs = sorted(pairs, key=lambda p: p[0])
    total = sum(w for _, w in pairs)
    if total <= 0:
        return None
    target = q * total
    acc = 0.0
    for value, w in pairs:
        acc += w
        if acc >= target:
            return value
    return pairs[-1][0]


def _region_metrics(name, cells, results, duration_s, target_gap_s, n_steps):
    idxs = [c.index for c in cells]
    wsum = sum(results[i].weight for i in idxs)
    if wsum <= 0:
        return None

    def wfrac(predicate):
        hit = sum(results[i].weight for i in idxs if predicate(results[i]))
        return hit / wsum

    ever = wfrac(lambda r: r.ever_visible)
    continuous = wfrac(lambda r: r.continuous)
    mean_inst = (
        sum(results[i].weight * results[i].visible_steps / n_steps for i in idxs) / wsum
    )
    # Interior gaps exclude edge waits; the service gap includes them.
    interior_pairs = [
        (results[i].effective_worst_gap_s, results[i].weight)
        for i in idxs
        if results[i].effective_worst_gap_s is not None
    ]
    outage_pairs = [
        (results[i].worst_outage_s, results[i].weight)
        for i in idxs
        if results[i].worst_outage_s is not None
    ]
    mean_gap_pairs = [
        (results[i].mean_gap_s, results[i].weight)
        for i in idxs
        if results[i].mean_gap_s is not None
    ]
    unresolved = [
        i for i in idxs
        if results[i].ever_visible and results[i].effective_worst_gap_s is None
    ]
    served = 0.0
    if target_gap_s is not None:
        served = wfrac(
            lambda r: r.effective_worst_gap_s is not None
            and r.effective_worst_gap_s <= target_gap_s
        )
    initial = [results[i].first_access_s for i in idxs
               if results[i].first_access_s is not None and not results[i].continuous]
    tails = [duration_s - results[i].last_access_s for i in idxs
             if results[i].last_access_s is not None and not results[i].continuous]
    folds = [results[i].min_fold for i in idxs if results[i].ever_visible]
    unserved = sum(1 for i in idxs if not results[i].ever_visible)
    return RegionMetrics(
        region=name,
        cell_count=len(idxs),
        weight_fraction=wsum,
        ever_coverage=ever,
        continuous_coverage=continuous,
        mean_instantaneous_coverage=mean_inst,
        worst_interior_gap_s=max((v for v, _ in outage_pairs), default=None),
        worst_gap_interior_only_s=max((v for v, _ in interior_pairs), default=None),
        p90_interior_gap_s=_weighted_quantile(interior_pairs, 0.9) if interior_pairs else None,
        mean_interior_gap_s=(sum(v * w for v, w in mean_gap_pairs) / sum(w for _, w in mean_gap_pairs)) if mean_gap_pairs else None,
        worst_initial_wait_s=max(initial, default=None),
        worst_tail_gap_s=max(tails, default=None),
        served_on_cadence=served,
        min_fold=min(folds, default=None),
        gap_revisit_target_s=target_gap_s,
        unserved_cells=unserved,
        unresolved_cadence_cells=len(unresolved),
        unresolved_cadence_fraction=sum(results[i].weight for i in unresolved) / wsum,
    )


def _merge_region_accross_epochs(regions, target_gap_s):
    """Combine per-epoch RegionMetrics into a worst-case envelope.

    Coverage uses the minimum across epochs; worst waits take the maximum;
    cells with unresolved cadence in any epoch keep that status.
    """
    base = regions[0]
    return RegionMetrics(
        region=base.region,
        cell_count=base.cell_count,
        weight_fraction=base.weight_fraction,
        ever_coverage=min(r.ever_coverage for r in regions),
        continuous_coverage=min(r.continuous_coverage for r in regions),
        mean_instantaneous_coverage=sum(r.mean_instantaneous_coverage for r in regions) / len(regions),
        worst_interior_gap_s=max((r.worst_interior_gap_s for r in regions), default=None),
        worst_gap_interior_only_s=max((r.worst_gap_interior_only_s for r in regions), default=None),
        p90_interior_gap_s=max((r.p90_interior_gap_s for r in regions if r.p90_interior_gap_s is not None), default=None),
        mean_interior_gap_s=max((r.mean_interior_gap_s for r in regions if r.mean_interior_gap_s is not None), default=None),
        worst_initial_wait_s=max((r.worst_initial_wait_s for r in regions if r.worst_initial_wait_s is not None), default=None),
        worst_tail_gap_s=max((r.worst_tail_gap_s for r in regions if r.worst_tail_gap_s is not None), default=None),
        served_on_cadence=min(r.served_on_cadence for r in regions),
        min_fold=min((r.min_fold for r in regions if r.min_fold is not None), default=None),
        gap_revisit_target_s=target_gap_s,
        unserved_cells=max(r.unserved_cells for r in regions),
        unresolved_cadence_cells=max(r.unresolved_cadence_cells for r in regions),
        unresolved_cadence_fraction=max(r.unresolved_cadence_fraction for r in regions),
    )


def evaluate_pattern(
    pattern: ConstellationPattern,
    duration_s: float,
    step_s: float,
    min_elevation_rad: float = 0.0,
    grid_lat_step_deg: float = 5.0,
    grid_lon_step_deg: float = 5.0,
    revisit_target_s: Optional[float] = None,
    polar_latitude_deg: float = 75.0,
    epoch_s: float = 0.0,
    cells: Optional[list[GridCell]] = None,
    phase_offsets_s: Optional[list[float]] = None,
) -> ServiceMetrics:
    """Instantiate, propagate and score one constellation pattern.

    With ``phase_offsets_s`` the whole window is repeated at several constellation
    phases; reported gaps are the worst-case envelope, removing dependence on the
    arbitrary simulation epoch (important for Walker seam gaps).
    """
    if duration_s <= 0 or step_s <= 0:
        raise ValueError("duration and step must be positive")
    if step_s > duration_s:
        raise ValueError("step must not exceed duration")
    offsets = [0.0] if phase_offsets_s is None else [0.0, *phase_offsets_s]
    design = instantiate(pattern, epoch_s=epoch_s)
    if cells is None:
        cells = area_weighted_grid(grid_lat_step_deg, grid_lon_step_deg)
    alpha = access_half_angle_rad(pattern.alt_km, min_elevation_rad)

    all_metrics = []
    for off in offsets:
        sampler = ConstellationSampler(design, phase_offset_s=off)
        results, n_steps = _simulate_cells(
            sampler, cells, math.cos(alpha), math.sin(alpha), duration_s, step_s
        )
        polar_lat = math.radians(polar_latitude_deg)
        north = [c for c in cells if c.lat_rad >= polar_lat]
        south = [c for c in cells if c.lat_rad <= -polar_lat]
        polar_cells = list({c.index: c for c in north + south}.values())
        all_metrics.append({
            "globe": _region_metrics("globe", cells, results, duration_s, revisit_target_s, n_steps),
            "north": _region_metrics("north", north, results, duration_s, revisit_target_s, n_steps),
            "south": _region_metrics("south", south, results, duration_s, revisit_target_s, n_steps),
            "polar": _region_metrics("polar", polar_cells, results, duration_s, revisit_target_s, n_steps),
        })

    globe = _merge_region_accross_epochs([m["globe"] for m in all_metrics], revisit_target_s)
    nm = _merge_region_accross_epochs([m["north"] for m in all_metrics], revisit_target_s)
    sm = _merge_region_accross_epochs([m["south"] for m in all_metrics], revisit_target_s)
    pm = _merge_region_accross_epochs([m["polar"] for m in all_metrics], revisit_target_s)
    nm.region = f"north_polar_{polar_latitude_deg:g}"
    sm.region = f"south_polar_{polar_latitude_deg:g}"
    pm.region = f"polar_{polar_latitude_deg:g}"

    last_results = results if len(offsets) == 1 else []

    return ServiceMetrics(
        pattern_id=pattern.pattern_id,
        satellites=pattern.satellites,
        duration_s=duration_s,
        step_s=step_s,
        grid_lat_step_deg=grid_lat_step_deg,
        grid_lon_step_deg=grid_lon_step_deg,
        min_elevation_rad=min_elevation_rad,
        footprint_half_angle_rad=alpha,
        period_s=design.period_s,
        cells=last_results,
        globe=globe,
        north_polar=nm,
        south_polar=sm,
        polar=pm,
    )
