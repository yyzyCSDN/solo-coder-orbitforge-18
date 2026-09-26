"""Constellation designer: turn orbital shells and Walker patterns into concrete
satellites, simulate global / polar coverage and revisit, and search for
combinations that satisfy explicit service requirements.

The gate is *not* a mean coverage number. Every candidate is assessed against
hard limits on global coverage, polar coverage and worst-case revisit gap, and
candidates that miss any limit are rejected with the offending metric reported.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional

from orbitforge.core.constants import (
    MU_EARTH_KM3_S2,
    R_EARTH_EQUATOR_KM,
    OMEGA_EARTH_RAD_S,
)
from orbitforge.coverage.constellation import walker_delta
from orbitforge.orbits.elements import KeplerianElements


# ---------------------------------------------------------------------------
# Building blocks: shells, patterns, individual satellites
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrbitalShell:
    """A circular orbital shell shared by a group of satellites."""
    name: str
    altitude_km: float
    inclination_deg: float
    eccentricity: float = 0.0
    argp_deg: float = 0.0

    @property
    def semi_major_axis_km(self) -> float:
        return R_EARTH_EQUATOR_KM + self.altitude_km

    @property
    def period_s(self) -> float:
        return 2.0 * math.pi * math.sqrt(self.semi_major_axis_km ** 3 / MU_EARTH_KM3_S2)


@dataclass(frozen=True)
class WalkerPattern:
    """Walker delta pattern: T satellites, P planes, F phasing parameter."""
    total: int
    planes: int
    phasing: int

    def validate(self) -> None:
        if self.total <= 0 or self.planes <= 0:
            raise ValueError('total satellites and planes must be positive')
        if self.total % self.planes != 0:
            raise ValueError('satellites must divide evenly among planes')
        if not 0 <= self.phasing < self.planes:
            raise ValueError('phasing parameter F must satisfy 0 <= F < P')


@dataclass(frozen=True)
class SatelliteSlot:
    """One concrete satellite: its Walker slot and full orbital parameters."""
    sat_id: str
    plane: int
    slot_in_plane: int
    altitude_km: float
    inclination_rad: float
    raan_rad: float
    mean_anomaly_rad: float
    argp_rad: float
    eccentricity: float
    semi_major_axis_km: float
    period_s: float

    def keplerian_elements(self) -> KeplerianElements:
        # Circular orbit: true anomaly equals mean anomaly.
        return KeplerianElements(
            a_km=self.semi_major_axis_km,
            e=self.eccentricity,
            i_rad=self.inclination_rad,
            raan_rad=self.raan_rad,
            argp_rad=self.argp_rad,
            nu_rad=self.mean_anomaly_rad,
        )


def build_shell_satellites(shell: OrbitalShell, pattern: WalkerPattern) -> list[SatelliteSlot]:
    """Instantiate every satellite of a Walker delta shell with slot and orbit."""
    pattern.validate()
    if shell.altitude_km <= 0:
        raise ValueError('altitude must be positive')
    slots = walker_delta(pattern.total, pattern.planes, pattern.phasing)
    inc = math.radians(shell.inclination_deg)
    argp = math.radians(shell.argp_deg)
    out: list[SatelliteSlot] = []
    for ws in slots:
        sat_id = f'{shell.name}-P{ws.plane + 1:02d}-S{ws.slot + 1:02d}'
        out.append(SatelliteSlot(
            sat_id=sat_id,
            plane=ws.plane,
            slot_in_plane=ws.slot,
            altitude_km=shell.altitude_km,
            inclination_rad=inc,
            raan_rad=ws.raan_rad % (2.0 * math.pi),
            mean_anomaly_rad=ws.mean_anomaly_rad % (2.0 * math.pi),
            argp_rad=argp,
            eccentricity=shell.eccentricity,
            semi_major_axis_km=shell.semi_major_axis_km,
            period_s=shell.period_s,
        ))
    return out


def max_central_angle_rad(alt_km: float, min_elevation_rad: float = 0.0) -> float:
    """Geocentric angular radius of the footprint down to min elevation."""
    re = R_EARTH_EQUATOR_KM
    rs = re + alt_km
    return math.acos(re / rs * math.cos(min_elevation_rad)) - min_elevation_rad


# ---------------------------------------------------------------------------
# Service requirements and measured metrics
# ---------------------------------------------------------------------------

@dataclass
class ServiceRequirements:
    horizon_s: float = 86164.0                 # one sidereal day
    step_s: float = 120.0
    epoch_tai_s: float = 0.0
    min_elevation_deg: float = 10.0
    grid_lat_step_deg: float = 10.0
    grid_lon_step_deg: float = 10.0
    polar_latitude_deg: float = 66.5
    min_global_coverage: float = 0.98
    min_polar_coverage: float = 0.90
    max_revisit_gap_s: float = 4.0 * 3600.0


@dataclass
class ServiceMetrics:
    n_cells: int
    n_steps: int
    step_s: float
    horizon_s: float
    global_coverage: float                    # area-weighted, whole horizon
    polar_coverage: float                     # area-weighted, |lat| >= polar
    mean_instantaneous_coverage: float        # average fold of the grid per step
    mean_fold: float                          # average access multiplicity
    worst_revisit_gap_s: Optional[float]      # max wait, incl. horizon edges
    p90_revisit_gap_s: Optional[float]
    revisit_compliant_fraction: float         # cells whose worst gap <= limit
    uncovered_cells: int


@dataclass
class GateCheck:
    name: str
    passed: bool
    actual: Optional[float]
    limit: float
    description: str


@dataclass
class CandidateResult:
    altitude_km: float
    inclination_deg: float
    total: int
    planes: int
    phasing: int
    metrics: ServiceMetrics
    passed: bool
    checks: list[GateCheck]

    @property
    def failures(self) -> list[str]:
        return [c.description for c in self.checks if not c.passed]


@dataclass
class DesignReport:
    requirements: dict
    selected: Optional[CandidateResult]
    satellites: list[SatelliteSlot] = field(default_factory=list)
    feasible: list[CandidateResult] = field(default_factory=list)
    rejected: list[CandidateResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Grid and time simulation
# ---------------------------------------------------------------------------

@dataclass
class _Grid:
    lat_rad: list[float]
    lon_rad: list[float]
    cell_weight: list[float]                # area weight cos(lat), per cell
    polar: bytearray                        # per-cell polar mask
    total_weight: float
    polar_weight: float


def _build_grid(req: ServiceRequirements) -> _Grid:
    dlat = math.radians(req.grid_lat_step_deg)
    dlon = math.radians(req.grid_lon_step_deg)
    lats: list[float] = []
    lat = -math.pi / 2.0 + dlat / 2.0
    while lat < math.pi / 2.0:
        lats.append(lat)
        lat += dlat
    lons: list[float] = []
    lon = -math.pi + dlon / 2.0
    while lon < math.pi:
        lons.append(lon)
        lon += dlon
    polar_lat = math.radians(req.polar_latitude_deg)
    weights: list[float] = []
    polar = bytearray(len(lats) * len(lons))
    total_w = 0.0
    polar_w = 0.0
    for band, phi in enumerate(lats):
        w = max(0.0, math.cos(phi))
        for k in range(len(lons)):
            idx = band * len(lons) + k
            weights.append(w)
            total_w += w
            if abs(phi) >= polar_lat:
                polar[idx] = 1
                polar_w += w
    return _Grid(lats, lons, weights, polar, total_w, polar_w)


def evaluate_shell(
    satellites: list[SatelliteSlot],
    req: ServiceRequirements,
    grid: Optional[_Grid] = None,
) -> ServiceMetrics:
    """Propagate every satellite over the horizon and score service metrics.

    A cell is "covered" if at least one satellite rises above the minimum
    elevation at any sampled time. Revisit gaps are derived per cell from its
    access timeline; the gaps at the two edges of the horizon (wait to first
    access, wait after last access) are counted, so the metric is a worst-case
    wait for service rather than an optimistic inter-pass average.
    """
    if not satellites:
        raise ValueError('cannot evaluate an empty satellite list')
    if grid is None:
        grid = _build_grid(req)
    nlat = len(grid.lat_rad)
    nlon = len(grid.lon_rad)
    ncell = nlat * nlon
    nsteps = max(1, int(round(req.horizon_s / req.step_s)))
    dt = req.horizon_s / nsteps

    # access[cell] is a timeline bytearray.
    access = [bytearray(nsteps) for _ in range(ncell)]

    theta = max_central_angle_rad(satellites[0].altitude_km,
                                  math.radians(req.min_elevation_deg))
    altitudes = {s.altitude_km for s in satellites}
    if len(altitudes) != 1:
        raise ValueError('evaluate_shell scores one common-altitude shell at a time')
    cos_theta = math.cos(theta)
    dlon = 2.0 * math.pi / nlon
    sin_lat = [math.sin(p) for p in grid.lat_rad]
    cos_lat = [math.cos(p) for p in grid.lat_rad]

    # Earth rotation angle advances linearly from the epoch GMST value.
    gmst0 = _gmst(req.epoch_tai_s)
    mark_count = 0

    for sat in satellites:
        a = sat.semi_major_axis_km
        n_motion = math.sqrt(MU_EARTH_KM3_S2 / a ** 3)
        ci, si = math.cos(sat.inclination_rad), math.sin(sat.inclination_rad)
        co, so = math.cos(sat.raan_rad), math.sin(sat.raan_rad)
        m0 = sat.mean_anomaly_rad
        for j in range(nsteps):
            t = j * dt
            m = m0 + n_motion * t
            cm, sm = math.cos(m), math.sin(m)
            # Position in ECI for a circular orbit: Rz(RAAN) Rx(i) r_perifocal.
            x = a * (cm * co - sm * ci * so)
            y = a * (cm * so + sm * ci * co)
            z = a * sm * si
            g = gmst0 + OMEGA_EARTH_RAD_S * t
            cg, sg = math.cos(g), math.sin(g)
            xe = x * cg + y * sg
            ye = -x * sg + y * cg
            phi_s = math.atan2(z, math.hypot(xe, ye))
            lam_s = math.atan2(ye, xe)

            sp, cp = math.sin(phi_s), math.cos(phi_s)
            newly: list[int] = []
            for band in range(nlat):
                dphi = abs(grid.lat_rad[band] - phi_s)
                if dphi > theta and abs(dphi - 2.0 * math.pi) > theta:
                    continue
                denom = cp * cos_lat[band]
                rhs = (cos_theta - sp * sin_lat[band]) / denom if denom > 1e-12 else 1.0
                if rhs <= -1.0:
                    continue
                half = math.pi if rhs >= 1.0 else math.acos(rhs)
                k0 = int(round(((lam_s + math.pi) % (2.0 * math.pi)) / dlon)) % nlon
                span = int(half / dlon) + 1
                for dk in range(-span, span + 1):
                    k = (k0 + dk) % nlon
                    idx = band * nlon + k
                    if access[idx][j]:
                        continue
                    dlam = abs(grid.lon_rad[k] - lam_s)
                    if dlam > math.pi:
                        dlam = 2.0 * math.pi - dlam
                    if dlam <= half + 1e-12:
                        access[idx][j] = 1
                        newly.append(idx)
            mark_count += len(newly)

    # Instantaneous (fold-of-grid) coverage per step, derived from the
    # boolean access matrix so each cell counts once no matter how many
    # satellites see it.
    inst_weight_sum = 0.0
    for j in range(nsteps):
        for idx in range(ncell):
            if access[idx][j]:
                inst_weight_sum += grid.cell_weight[idx]

    # Per-cell coverage and gap statistics.
    covered_w = 0.0
    polar_cov_w = 0.0
    uncovered = 0
    worst_gaps: list[float] = []
    for idx in range(ncell):
        line = access[idx]
        first = line.find(1)
        if first < 0:
            uncovered += 1
            continue
        covered_w += grid.cell_weight[idx]
        if grid.polar[idx]:
            polar_cov_w += grid.cell_weight[idx]
        last = line.rfind(1)
        worst = max(first * dt, (nsteps - 1 - last) * dt)
        run = 0
        for j in range(first, last + 1):
            if line[j]:
                if run:
                    worst = max(worst, run * dt)
                    run = 0
            else:
                run += 1
        if run:
            worst = max(worst, run * dt)
        worst_gaps.append(worst)

    global_cov = covered_w / grid.total_weight if grid.total_weight else 0.0
    polar_cov = polar_cov_w / grid.polar_weight if grid.polar_weight else 1.0
    mean_inst = (inst_weight_sum / nsteps) / grid.total_weight if grid.total_weight else 0.0
    mean_fold = mark_count / (ncell * nsteps)
    if worst_gaps:
        ordered = sorted(worst_gaps)
        p90 = ordered[min(len(ordered) - 1, int(math.ceil(0.9 * len(ordered)) - 1))]
        worst_gap = ordered[-1]
        compliant = sum(1 for g in worst_gaps if g <= req.max_revisit_gap_s) / len(worst_gaps)
    else:
        p90 = None
        worst_gap = None
        compliant = 0.0

    return ServiceMetrics(
        n_cells=ncell,
        n_steps=nsteps,
        step_s=dt,
        horizon_s=req.horizon_s,
        global_coverage=global_cov,
        polar_coverage=polar_cov,
        mean_instantaneous_coverage=mean_inst,
        mean_fold=mean_fold,
        worst_revisit_gap_s=worst_gap,
        p90_revisit_gap_s=p90,
        revisit_compliant_fraction=compliant,
        uncovered_cells=uncovered,
    )


def _gmst(tai_s: float) -> float:
    # Local import keeps the designer usable from pure geometry contexts and
    # avoids paying time-scale conversion cost per time step.
    from orbitforge.time.sidereal import gmst_angle
    return gmst_angle(tai_s)


# ---------------------------------------------------------------------------
# Gating and search
# ---------------------------------------------------------------------------

def assess(metrics: ServiceMetrics, req: ServiceRequirements) -> list[GateCheck]:
    checks = [
        GateCheck(
            'global_coverage',
            metrics.global_coverage + 1e-12 >= req.min_global_coverage,
            metrics.global_coverage, req.min_global_coverage,
            f'global coverage {metrics.global_coverage:.3f} < {req.min_global_coverage:.3f}',
        ),
        GateCheck(
            'polar_coverage',
            metrics.polar_coverage + 1e-12 >= req.min_polar_coverage,
            metrics.polar_coverage, req.min_polar_coverage,
            f'polar coverage {metrics.polar_coverage:.3f} < {req.min_polar_coverage:.3f}',
        ),
    ]
    gap = metrics.worst_revisit_gap_s
    gap_ok = gap is not None and gap <= req.max_revisit_gap_s + 1e-9
    checks.append(GateCheck(
        'worst_revisit_gap',
        gap_ok,
        gap, req.max_revisit_gap_s,
        (f'worst revisit gap {gap:.0f} s > {req.max_revisit_gap_s:.0f} s'
         if gap is not None else 'no access anywhere: revisit undefined'),
    ))
    return checks


def design_search(
    shell_name: str,
    altitudes_km: list[float],
    inclinations_deg: list[float],
    total_satellites: list[int],
    plane_options: list[int],
    req: ServiceRequirements,
    phasing_options: Optional[list[int]] = None,
) -> DesignReport:
    """Enumerate Walker candidates and keep those passing every service gate.

    The selected design is the passing candidate with the fewest satellites
    (ties broken by better global coverage, polar coverage and smaller gap).
    Every evaluated candidate is returned: ``feasible`` designs and, crucially,
    ``rejected`` designs carrying the exact gate(s) they failed.
    """
    grid = _build_grid(req)
    feasible: list[CandidateResult] = []
    rejected: list[CandidateResult] = []

    for alt in altitudes_km:
        for inc in inclinations_deg:
            shell = OrbitalShell(shell_name, alt, inc)
            for total in sorted(set(total_satellites)):
                for planes in sorted(set(plane_options)):
                    if total % planes != 0:
                        continue
                    f_values = (phasing_options if phasing_options is not None
                                else range(planes))
                    for f in f_values:
                        if not 0 <= f < planes:
                            continue
                        pattern = WalkerPattern(total, planes, f)
                        sats = build_shell_satellites(shell, pattern)
                        metrics = evaluate_shell(sats, req, grid)
                        checks = assess(metrics, req)
                        result = CandidateResult(
                            altitude_km=alt,
                            inclination_deg=inc,
                            total=total,
                            planes=planes,
                            phasing=f,
                            metrics=metrics,
                            passed=all(c.passed for c in checks),
                            checks=checks,
                        )
                        (feasible if result.passed else rejected).append(result)

    feasible.sort(key=lambda c: (c.total, -c.metrics.global_coverage,
                                 -c.metrics.polar_coverage,
                                 c.metrics.worst_revisit_gap_s or math.inf,
                                 c.altitude_km, c.inclination_deg))
    selected = feasible[0] if feasible else None
    satellites: list[SatelliteSlot] = []
    if selected is not None:
        shell = OrbitalShell(shell_name, selected.altitude_km,
                             selected.inclination_deg)
        pattern = WalkerPattern(selected.total, selected.planes,
                                selected.phasing)
        satellites = build_shell_satellites(shell, pattern)

    return DesignReport(
        requirements=asdict(req),
        selected=selected,
        satellites=satellites,
        feasible=feasible,
        rejected=rejected,
    )


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------

def _metrics_dict(m: ServiceMetrics) -> dict:
    return asdict(m)


def _candidate_dict(c: CandidateResult) -> dict:
    return {
        'altitude_km': c.altitude_km,
        'inclination_deg': c.inclination_deg,
        'walker': {'T': c.total, 'P': c.planes, 'F': c.phasing},
        'passed': c.passed,
        'failures': c.failures,
        'checks': [asdict(ch) for ch in c.checks],
        'metrics': _metrics_dict(c.metrics),
    }


def _satellite_dict(s: SatelliteSlot) -> dict:
    return {
        'sat_id': s.sat_id,
        'plane': s.plane,
        'slot_in_plane': s.slot_in_plane,
        'altitude_km': s.altitude_km,
        'semi_major_axis_km': s.semi_major_axis_km,
        'inclination_deg': math.degrees(s.inclination_rad),
        'raan_deg': math.degrees(s.raan_rad),
        'mean_anomaly_deg': math.degrees(s.mean_anomaly_rad),
        'argp_deg': math.degrees(s.argp_rad),
        'eccentricity': s.eccentricity,
        'period_min': s.period_s / 60.0,
    }


def report_to_dict(report: DesignReport) -> dict:
    return {
        'requirements': report.requirements,
        'selected': _candidate_dict(report.selected) if report.selected else None,
        'satellites': [_satellite_dict(s) for s in report.satellites],
        'n_feasible': len(report.feasible),
        'n_rejected': len(report.rejected),
        'feasible': [_candidate_dict(c) for c in report.feasible],
        'rejected': [_candidate_dict(c) for c in report.rejected],
    }
