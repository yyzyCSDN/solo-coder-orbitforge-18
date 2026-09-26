"""Tests for the constellation designer (run under pytest or standalone).

Covers: Walker slot geometry, propagation, per-cell coverage/revisit metrics,
polar service, explicit rejection of non-compliant combinations, and the end
to end search.
"""
from __future__ import annotations

import math

from orbitforge.constellation import (
    ConstellationPattern,
    ConstellationSampler,
    DesignSpace,
    SearchConfig,
    ServiceRequirements,
    area_weighted_grid,
    evaluate_pattern,
    geometric_prescreen,
    instantiate,
    search_design_space,
    verify,
)
from orbitforge.constellation.report import plan_to_dict, render_plan, satellite_rows

DEG = 180.0 / math.pi


def test_walker_slots_and_satellite_records():
    p = ConstellationPattern(satellites=12, planes=3, phasing=1, alt_km=550.0,
                             inclination_rad=math.radians(53.0))
    design = instantiate(p)
    assert design.total == 12
    rows = satellite_rows(design)
    assert len(rows) == 12
    # RAAN uniformly spaced; every satellite in plane 0 has RAAN 0.
    plane0_raans = {round(r["raan_deg"] % 120.0, 6) for r in rows if r["plane"] == 0}
    assert plane0_raans == {0.0}
    plane1 = [r for r in rows if r["plane"] == 1]
    plane0 = [r for r in rows if r["plane"] == 0]
    # F=1 shifts plane 1 anomalies by 360*F/T = 30 deg relative to plane 0.
    for r0, r1 in zip(sorted(plane0, key=lambda r: r["slot"]),
                      sorted(plane1, key=lambda r: r["slot"])):
        assert abs(((r1["mean_anomaly_deg"] - r0["mean_anomaly_deg"]) % 360.0) - 30.0) < 1e-9
    # Every record carries full orbital parameters.
    for r in rows:
        assert r["alt_km"] == 550.0
        assert abs(r["i_deg"] - 53.0) < 1e-9
        assert r["period_min"] > 90.0


def test_invalid_pattern_rejected():
    for kw in (
        dict(satellites=10, planes=3, phasing=0, alt_km=600.0, inclination_rad=1.0),
        dict(satellites=12, planes=3, phasing=9, alt_km=600.0, inclination_rad=1.0),
    ):
        try:
            ConstellationPattern(**kw)
            assert False, "expected ValueError"
        except ValueError:
            pass


def test_grid_area_weights_and_propagation():
    cells = area_weighted_grid(10.0, 10.0)
    assert abs(sum(c.weight for c in cells) - 1.0) < 1e-9
    # One equatorial satellite never leaves the equator.
    p = ConstellationPattern(1, 1, 0, 550.0, 0.0)
    sampler = ConstellationSampler(instantiate(p))
    for dt in (0.0, 300.0, 3000.0):
        lat, _ = sampler.sample(dt)[0]
        assert abs(lat) < 1e-12


def test_sparse_constellation_rejected_on_cadence_not_just_mean_coverage():
    # 24 satellites in 3 planes see the whole globe over 30 h but leave long
    # waits: it must be REJECTED on the gap criteria even though ever coverage
    # is 100% (a mean-only metric would wrongly pass it).
    p = ConstellationPattern(24, 3, 1, 600.0, math.radians(97.8))
    m = evaluate_pattern(
        p, duration_s=30 * 3600.0, step_s=300.0,
        min_elevation_rad=math.radians(10.0),
        grid_lat_step_deg=10.0, grid_lon_step_deg=10.0,
        revisit_target_s=2 * 3600.0, polar_latitude_deg=75.0,
    )
    assert m.globe.ever_coverage == 1.0
    assert m.polar.ever_coverage == 1.0
    assert m.globe.worst_interior_gap_s > 2 * 3600.0
    req = ServiceRequirements.degrees(
        min_elevation_deg=10.0, max_global_gap_s=2 * 3600.0,
        max_polar_gap_s=2 * 3600.0,
    )
    v = verify(m, req)
    assert v.accepted is False
    names = {c.name for c in v.criteria if not c.passed}
    assert "global_revisit_gap" in names
    assert all("coverage" not in n for n in names)
    assert v.rejection_reasons  # explicit reasons, not a silent mean-only pass


def test_equatorial_pattern_prescreened_for_polar_requirement():
    req = ServiceRequirements.degrees(min_elevation_deg=10.0)
    p = ConstellationPattern(48, 6, 1, 800.0, 0.0)
    reason = geometric_prescreen(p, req)
    assert reason is not None and "polar" in reason
    p2 = ConstellationPattern(48, 6, 1, 800.0, math.radians(90.0))
    assert geometric_prescreen(p2, req) is None


def test_continuous_constellation_passes_end_to_end():
    # 70 sats in 7 planes at 1000 km / 83 deg gives global continuous
    # coverage above 10 deg elevation: worst wait is zero everywhere.
    p = ConstellationPattern(70, 7, 5, 1000.0, math.radians(83.0))
    req = ServiceRequirements.degrees(
        min_elevation_deg=10.0,
        max_global_gap_s=2 * 3600.0,
        max_polar_gap_s=2 * 3600.0,
        polar_latitude_deg=75.0,
    )
    m = evaluate_pattern(
        p, duration_s=24 * 3600.0, step_s=300.0,
        min_elevation_rad=math.radians(10.0),
        grid_lat_step_deg=10.0, grid_lon_step_deg=10.0,
        revisit_target_s=2 * 3600.0, polar_latitude_deg=75.0,
        phase_offsets_s=[_period_fraction(p, 0.25), _period_fraction(p, 0.5),
                         _period_fraction(p, 0.75)],
    )
    v = verify(m, req)
    assert m.globe.ever_coverage == 1.0
    assert m.polar.ever_coverage == 1.0
    assert m.globe.continuous_coverage == 1.0
    assert v.accepted, v.rejection_reasons
    plan = plan_to_dict(p, m, v)
    assert len(plan["satellites"]) == 70
    assert plan["verdict"]["accepted"] is True
    text = render_plan(p, m, v)
    assert "S07-10" in text


def _period_fraction(pattern, fraction):
    from orbitforge.core.constants import R_EARTH_EQUATOR_KM
    from orbitforge.orbits.periods import orbital_period_s
    return fraction * orbital_period_s(R_EARTH_EQUATOR_KM + pattern.alt_km)


def test_multi_epoch_does_not_depend_on_start_phase():
    # The 70-sat continuous design must pass at every phase offset; a sparse
    # design must fail at at least one of them (Walker seam exposed).
    req = ServiceRequirements.degrees(
        min_elevation_deg=10.0, max_global_gap_s=2 * 3600.0,
        max_polar_gap_s=2 * 3600.0,
    )
    p = ConstellationPattern(70, 7, 5, 1000.0, math.radians(83.0))
    m = evaluate_pattern(
        p, duration_s=20 * 3600.0, step_s=300.0,
        min_elevation_rad=math.radians(10.0),
        grid_lat_step_deg=10.0, grid_lon_step_deg=10.0,
        revisit_target_s=2 * 3600.0,
        phase_offsets_s=[_period_fraction(p, f) for f in (0.13, 0.37, 0.61, 0.88)],
    )
    assert verify(m, req).accepted


def test_search_finds_and_ranks_and_keeps_rejections():
    space = DesignSpace(
        altitudes_km=[1000.0],
        inclinations_deg=[0.0, 83.0],
        planes=[7],
        per_plane=[10],
        phasings=[5],
    )
    req = ServiceRequirements.degrees(
        min_elevation_deg=10.0,
        max_global_gap_s=2 * 3600.0,
        max_polar_gap_s=2 * 3600.0,
    )
    cfg = SearchConfig(
        duration_s=24 * 3600.0, step_s=300.0,
        grid_lat_step_deg=10.0, grid_lon_step_deg=10.0,
        phase_offset_fractions=(0.25, 0.5, 0.75),
    )
    report = search_design_space(space, req, cfg)
    # Equatorial family geometry-rejected; 70-sat polar family accepted.
    assert len(report.accepted) == 1
    assert any(c.rejection_stage == "geometric" for c in report.rejected)
    best = report.best()
    assert best.accepted and best.pattern.inclination_rad == math.radians(83.0)
    assert best.metrics.globe.continuous_coverage == 1.0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
