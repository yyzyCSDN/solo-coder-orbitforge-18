import math

import pytest
from fastapi.testclient import TestClient

from orbitforge.api.app import app
from orbitforge.constellation import (
    OrbitalShell,
    WalkerPattern,
    ServiceRequirements,
    build_shell_satellites,
    max_central_angle_rad,
    evaluate_shell,
    assess,
    design_search,
    report_to_dict,
)

# Coarse / short settings keep the search tests fast; the production defaults
# in ServiceRequirements use a sidereal day and a 10 deg grid.
REQ = ServiceRequirements(
    horizon_s=4 * 3600.0,
    step_s=300.0,
    grid_lat_step_deg=15.0,
    grid_lon_step_deg=15.0,
    min_elevation_deg=10.0,
    polar_latitude_deg=66.5,
    min_global_coverage=0.90,
    min_polar_coverage=0.90,
    max_revisit_gap_s=3600.0,
)


def test_d01_shell_period_and_semi_major_axis():
    shell = OrbitalShell('LEO', 600.0, 53.0)
    assert shell.semi_major_axis_km == pytest.approx(6378.137 + 600.0)
    # 600 km circular orbit period is about 96.7 minutes.
    assert 96.0 < shell.period_s / 60.0 < 97.5


def test_d02_walker_pattern_validation():
    with pytest.raises(ValueError):
        WalkerPattern(25, 8, 1)          # not divisible among planes
    with pytest.raises(ValueError):
        WalkerPattern(24, 8, 8)          # F must be < P
    with pytest.raises(ValueError):
        WalkerPattern(0, 0, 0)


def test_d03_central_angle_monotonic_with_altitude_and_elevation():
    low = max_central_angle_rad(500.0, 0.0)
    high = max_central_angle_rad(1200.0, 0.0)
    horizon = math.acos(6378.137 / (6378.137 + 1200.0))
    assert 0 < low < high <= horizon
    assert max_central_angle_rad(1200.0, math.radians(20.0)) < high


def test_d04_satellite_slots_carry_full_orbit_parameters():
    shell = OrbitalShell('LEO', 1200.0, 74.0)
    sats = build_shell_satellites(shell, WalkerPattern(24, 8, 4))
    assert len(sats) == 24
    p1 = [s for s in sats if s.plane == 0]
    p2 = [s for s in sats if s.plane == 1]
    assert len(p1) == 3 and len(p2) == 3
    # Plane spacing 360/8 = 45 deg; Walker F=4 adds 360*F/T = 60 deg phase.
    assert math.degrees(p2[0].raan_rad) == pytest.approx(45.0)
    assert math.degrees(p2[0].mean_anomaly_rad) == pytest.approx(60.0)
    # Every slot is usable as a full element set.
    for s in sats:
        el = s.keplerian_elements()
        assert el.a_km == pytest.approx(s.semi_major_axis_km)
        assert el.e == 0.0
        assert math.degrees(el.i_rad) == pytest.approx(74.0)


def test_d05_coverage_dense_constellation_passes():
    shell = OrbitalShell('LEO', 1200.0, 74.0)
    sats = build_shell_satellites(shell, WalkerPattern(24, 8, 1))
    m = evaluate_shell(sats, REQ)
    checks = assess(m, REQ)
    assert all(c.passed for c in checks), [c.description for c in checks if not c.passed]
    assert m.global_coverage >= 0.95
    assert m.polar_coverage >= 0.90
    assert m.worst_revisit_gap_s <= 3600.0
    assert 0.0 < m.mean_instantaneous_coverage <= 1.0


def test_d06_sparse_constellation_is_rejected_not_averaged_away():
    shell = OrbitalShell('LEO', 1200.0, 74.0)
    sats = build_shell_satellites(shell, WalkerPattern(4, 2, 0))
    m = evaluate_shell(sats, REQ)
    results = {c.name: c for c in assess(m, REQ)}
    assert not results['global_coverage'].passed
    assert m.global_coverage < 0.9


def test_d07_inclination_gates_polar_service():
    # A 53 deg shell can never reach the polar caps even with many sats.
    req = ServiceRequirements(
        horizon_s=4 * 3600.0, step_s=300.0,
        grid_lat_step_deg=15.0, grid_lon_step_deg=15.0,
        min_global_coverage=0.90, min_polar_coverage=0.90,
        max_revisit_gap_s=6 * 3600.0)
    shell = OrbitalShell('LEO', 1200.0, 53.0)
    sats = build_shell_satellites(shell, WalkerPattern(30, 6, 1))
    m = evaluate_shell(sats, req)
    results = {c.name: c for c in assess(m, req)}
    assert results['global_coverage'].passed
    assert not results['polar_coverage'].passed
    assert m.polar_coverage < 0.80


def test_d08_edge_gaps_count_as_revisit_wait():
    # One satellite, short horizon: first/last edge waits must be counted.
    shell = OrbitalShell('LEO', 1200.0, 74.0)
    sats = build_shell_satellites(shell, WalkerPattern(1, 1, 0))
    m = evaluate_shell(sats, REQ)
    assert m.worst_revisit_gap_s is not None
    assert m.worst_revisit_gap_s >= m.p90_revisit_gap_s
    assert m.worst_revisit_gap_s > 1800.0  # single sat cannot revisit hourly


def test_d09_design_search_reports_feasible_and_rejected():
    report = design_search(
        shell_name='LEO',
        altitudes_km=[1200.0],
        inclinations_deg=[53.0, 74.0],
        total_satellites=[4, 24],
        plane_options=[2, 8],
        phasing_options=[0, 1],
        req=REQ,
    )
    assert report.selected is not None
    assert report.selected.total == 24
    assert len(report.satellites) == 24
    # Rejected rows must say exactly which gate failed.
    assert report.rejected
    for cand in report.rejected:
        assert cand.passed is False
        assert cand.failures
    # 53 deg designs must all be rejected on the polar gate.
    for cand in report.feasible:
        assert cand.inclination_deg > 53.0
    payload = report_to_dict(report)
    assert payload['selected']['passed'] is True
    assert payload['n_rejected'] == len(report.rejected)
    assert {'sat_id', 'raan_deg', 'mean_anomaly_deg'} <= set(payload['satellites'][0])


def test_d10_design_search_can_find_no_solution():
    strict = ServiceRequirements(
        horizon_s=2 * 3600.0, step_s=300.0,
        grid_lat_step_deg=15.0, grid_lon_step_deg=15.0,
        min_global_coverage=1.0, min_polar_coverage=1.0,
        max_revisit_gap_s=60.0)
    report = design_search(
        shell_name='LEO',
        altitudes_km=[800.0], inclinations_deg=[53.0],
        total_satellites=[2], plane_options=[2],
        phasing_options=[0], req=strict)
    assert report.selected is None
    assert report.feasible == []
    assert len(report.rejected) == 1


def test_d11_evaluate_endpoint_reports_gates_and_slots():
    client = TestClient(app)
    r = client.post('/v1/constellation/evaluate', json={
        'shell_name': 'LEO', 'altitude_km': 1200.0, 'inclination_deg': 74.0,
        'total': 24, 'planes': 8, 'phasing': 1,
        'requirements': {
            'horizon_s': 4 * 3600.0, 'step_s': 300.0,
            'grid_lat_step_deg': 15.0, 'grid_lon_step_deg': 15.0,
            'min_global_coverage': 0.9, 'min_polar_coverage': 0.9,
            'max_revisit_gap_s': 3600.0,
        }})
    assert r.status_code == 200
    body = r.json()
    assert body['passed'] is True and body['failures'] == []
    assert len(body['satellites']) == 24
    assert body['satellites'][0]['sat_id'] == 'LEO-P01-S01'


def test_d12_design_endpoint_marks_failed_candidates():
    client = TestClient(app)
    r = client.post('/v1/constellation/design', json={
        'shell_name': 'LEO',
        'altitudes_km': [1200.0],
        'inclinations_deg': [53.0],
        'total_satellites': [4],
        'plane_options': [2],
        'phasing_options': [0],
        'requirements': {
            'horizon_s': 4 * 3600.0, 'step_s': 300.0,
            'grid_lat_step_deg': 15.0, 'grid_lon_step_deg': 15.0,
            'min_global_coverage': 0.98, 'min_polar_coverage': 0.95,
            'max_revisit_gap_s': 3600.0,
        }})
    assert r.status_code == 200
    body = r.json()
    assert body['selected'] is None
    assert body['n_feasible'] == 0 and body['n_rejected'] == 1
    failures = body['rejected'][0]['failures']
    assert any('polar coverage' in f for f in failures)
