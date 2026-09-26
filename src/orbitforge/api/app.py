from __future__ import annotations
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from orbitforge.core.vector import Vec3
from orbitforge.orbits.kepler import solve_kepler_elliptic
from orbitforge.maneuvers.hohmann import hohmann
from orbitforge.link.budget import free_space_loss_db
from orbitforge.environment.eclipse import eclipse_state
from orbitforge.attitude.quaternion import Quaternion
from orbitforge.storage.sqlite import Store
from orbitforge.constellation.designer import (
    OrbitalShell,
    WalkerPattern,
    ServiceRequirements,
    build_shell_satellites,
    evaluate_shell,
    assess,
    design_search,
    report_to_dict,
)
app = FastAPI(title='OrbitForge Mission Lab', version='1.0.0')
store = Store(':memory:')

class KeplerReq(BaseModel):
    mean_anomaly: float
    eccentricity: float

class HohmannReq(BaseModel):
    r1_km: float
    r2_km: float

class LinkReq(BaseModel):
    range_km: float
    freq_hz: float

class EclipseReq(BaseModel):
    sat: list[float]
    sun: list[float]

class RotateReq(BaseModel):
    q: list[float]
    v: list[float]


class ServiceRequirementsModel(BaseModel):
    horizon_s: float = 86164.0
    step_s: float = 120.0
    epoch_tai_s: float = 0.0
    min_elevation_deg: float = 10.0
    grid_lat_step_deg: float = 10.0
    grid_lon_step_deg: float = 10.0
    polar_latitude_deg: float = 66.5
    min_global_coverage: float = 0.98
    min_polar_coverage: float = 0.90
    max_revisit_gap_s: float = 4.0 * 3600.0


class EvaluateReq(BaseModel):
    shell_name: str = 'LEO'
    altitude_km: float
    inclination_deg: float
    total: int
    planes: int
    phasing: int
    requirements: ServiceRequirementsModel = ServiceRequirementsModel()


class DesignSearchReq(BaseModel):
    shell_name: str = 'LEO'
    altitudes_km: list[float]
    inclinations_deg: list[float]
    total_satellites: list[int]
    plane_options: list[int]
    phasing_options: Optional[list[int]] = None
    requirements: ServiceRequirementsModel = ServiceRequirementsModel()


def _requirements(m: ServiceRequirementsModel) -> ServiceRequirements:
    return ServiceRequirements(**m.model_dump())

@app.get('/live')
def live():
    return {'status': 'live'}

@app.get('/ready')
def ready():
    return {'status': 'ready'}

@app.post('/v1/orbit/kepler')
def kepler(r: KeplerReq):
    return {'eccentric_anomaly': solve_kepler_elliptic(r.mean_anomaly, r.eccentricity)}

@app.post('/v1/maneuver/hohmann')
def h(r: HohmannReq):
    return hohmann(r.r1_km, r.r2_km)

@app.post('/v1/link/fspl')
def l(r: LinkReq):
    return {'loss_db': free_space_loss_db(r.range_km, r.freq_hz)}

@app.post('/v1/environment/eclipse')
def e(r: EclipseReq):
    return {'state': eclipse_state(Vec3(*r.sat), Vec3(*r.sun))}

@app.post('/v1/attitude/rotate')
def rotate(r: RotateReq):
    q = Quaternion(*r.q)
    v = q.rotate(Vec3(*r.v))
    return {'v': v.as_tuple()}

@app.post('/v1/constellation/evaluate')
def evaluate_constellation(r: EvaluateReq):
    """Evaluate one Walker shell: per-satellite slots plus pass/fail gates."""
    req = _requirements(r.requirements)
    shell = OrbitalShell(r.shell_name, r.altitude_km, r.inclination_deg)
    pattern = WalkerPattern(r.total, r.planes, r.phasing)
    satellites = build_shell_satellites(shell, pattern)
    metrics = evaluate_shell(satellites, req)
    checks = assess(metrics, req)
    from dataclasses import asdict
    return {
        'walker': {'T': r.total, 'P': r.planes, 'F': r.phasing},
        'passed': all(c.passed for c in checks),
        'failures': [c.description for c in checks if not c.passed],
        'checks': [asdict(c) for c in checks],
        'metrics': asdict(metrics),
        'satellites': [
            {
                'sat_id': s.sat_id,
                'plane': s.plane,
                'slot_in_plane': s.slot_in_plane,
                'altitude_km': s.altitude_km,
                'semi_major_axis_km': s.semi_major_axis_km,
                'inclination_rad': s.inclination_rad,
                'raan_rad': s.raan_rad,
                'mean_anomaly_rad': s.mean_anomaly_rad,
                'argp_rad': s.argp_rad,
                'eccentricity': s.eccentricity,
                'period_s': s.period_s,
            }
            for s in satellites
        ],
    }


@app.post('/v1/constellation/design')
def design_constellation(r: DesignSearchReq):
    """Search Walker combinations and return the selected design plus every
    feasible and rejected candidate with the gate(s) it failed."""
    req = _requirements(r.requirements)
    report = design_search(
        shell_name=r.shell_name,
        altitudes_km=r.altitudes_km,
        inclinations_deg=r.inclinations_deg,
        total_satellites=r.total_satellites,
        plane_options=r.plane_options,
        req=req,
        phasing_options=r.phasing_options,
    )
    return report_to_dict(report)


@app.get('/v1/system/audit')
def audit():
    return store.audit_chain()

def main():
    import uvicorn
    uvicorn.run('orbitforge.api.app:app', host='127.0.0.1', port=8080)
