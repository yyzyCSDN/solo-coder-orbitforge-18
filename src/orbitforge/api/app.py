from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
from orbitforge.core.vector import Vec3
from orbitforge.orbits.kepler import solve_kepler_elliptic
from orbitforge.maneuvers.hohmann import hohmann
from orbitforge.link.budget import free_space_loss_db
from orbitforge.environment.eclipse import eclipse_state
from orbitforge.attitude.quaternion import Quaternion
from orbitforge.storage.sqlite import Store
from orbitforge.constellation.design import ConstellationPattern
from orbitforge.constellation.evaluator import evaluate_pattern
from orbitforge.constellation.requirements import ServiceRequirements, verify
from orbitforge.constellation.report import plan_to_dict
from orbitforge.constellation.search import (
    DesignSpace, SearchConfig, search_design_space, refine_candidate,
)
from orbitforge.constellation.report import report_to_dict
import math
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

class ConstellationReq(BaseModel):
    satellites: int
    planes: int
    phasing: int
    alt_km: float
    inclination_deg: float
    eccentricity: float = 0.0
    min_elevation_deg: float = 0.0
    polar_latitude_deg: float = 75.0
    duration_hours: float = 30.0
    step_s: float = 240.0
    grid_deg: float = 7.5
    max_global_gap_hours: float | None = None
    max_polar_gap_hours: float | None = None
    min_global_coverage: float = 1.0
    min_polar_coverage: float = 1.0

class ConstellationSearchReq(BaseModel):
    altitudes_km: list[float]
    inclinations_deg: list[float]
    planes: list[int]
    per_plane: list[int]
    phasings: list[int] | None = None
    min_elevation_deg: float = 0.0
    polar_latitude_deg: float = 75.0
    duration_hours: float = 30.0
    step_s: float = 240.0
    grid_deg: float = 7.5
    max_global_gap_hours: float | None = None
    max_polar_gap_hours: float | None = None
    min_global_coverage: float = 1.0
    min_polar_coverage: float = 1.0
    refine_best: bool = False
    fine_duration_hours: float = 48.0
    fine_step_s: float = 120.0
    fine_grid_deg: float = 5.0

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

def _requirements(r):
    return ServiceRequirements.degrees(
        min_elevation_deg=r.min_elevation_deg,
        polar_latitude_deg=r.polar_latitude_deg,
        min_global_coverage=r.min_global_coverage,
        max_global_gap_s=(r.max_global_gap_hours * 3600.0 if r.max_global_gap_hours is not None else None),
        min_polar_coverage=r.min_polar_coverage,
        max_polar_gap_s=(r.max_polar_gap_hours * 3600.0 if r.max_polar_gap_hours is not None else None),
    )

@app.post('/v1/constellation/evaluate')
def constellation_evaluate(r: ConstellationReq):
    pattern = ConstellationPattern(
        satellites=r.satellites, planes=r.planes, phasing=r.phasing,
        alt_km=r.alt_km, inclination_rad=math.radians(r.inclination_deg),
        eccentricity=r.eccentricity,
    )
    req = _requirements(r)
    metrics = evaluate_pattern(
        pattern,
        duration_s=r.duration_hours * 3600.0,
        step_s=r.step_s,
        min_elevation_rad=math.radians(r.min_elevation_deg),
        grid_lat_step_deg=r.grid_deg,
        grid_lon_step_deg=r.grid_deg,
        revisit_target_s=req.max_global_gap_s,
        polar_latitude_deg=r.polar_latitude_deg,
    )
    verdict = verify(metrics, req)
    return {'plan': plan_to_dict(pattern, metrics, verdict)}

@app.post('/v1/constellation/search')
def constellation_search(r: ConstellationSearchReq):
    req = _requirements(r)
    space = DesignSpace(
        altitudes_km=r.altitudes_km,
        inclinations_deg=r.inclinations_deg,
        planes=r.planes,
        per_plane=r.per_plane,
        phasings=r.phasings,
    )
    config = SearchConfig(
        duration_s=r.duration_hours * 3600.0,
        step_s=r.step_s,
        grid_lat_step_deg=r.grid_deg,
        grid_lon_step_deg=r.grid_deg,
    )
    report = search_design_space(space, req, config)
    selected = None
    best = report.best()
    if best is not None and r.refine_best:
        fine = refine_candidate(
            best, req,
            duration_s=r.fine_duration_hours * 3600.0,
            step_s=r.fine_step_s,
            grid_lat_step_deg=r.fine_grid_deg,
            grid_lon_step_deg=r.fine_grid_deg,
        )
        selected = plan_to_dict(fine.pattern, fine.metrics, fine.verdict)
    elif best is not None:
        selected = plan_to_dict(best.pattern, best.metrics, best.verdict)
    return {'search': report_to_dict(report), 'selected_plan': selected}

@app.get('/v1/system/audit')
def audit():
    return store.audit_chain()

def main():
    import uvicorn
    uvicorn.run('orbitforge.api.app:app', host='127.0.0.1', port=8080)
