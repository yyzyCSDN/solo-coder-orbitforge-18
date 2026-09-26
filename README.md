# OrbitForge Mission Lab

OrbitForge is a Python library and small HTTP service for spacecraft mission
analysis: orbital mechanics, mission geometry and resource products.

## Quick start

```bash
python -m pip install -e '.[test]'
PYTHONPATH=src python -m pytest -q
orbitforge
```

The service listens on `127.0.0.1:8080` by default. `GET /live` and `GET /ready`
report service state, and the analysis endpoints live under `/v1/`.

## Constellation designer

Turn orbital planes, phasing and altitude into individual satellites, then
verify global coverage, worst-case revisit and polar service:

```python
from orbitforge.constellation import (
    ConstellationPattern, DesignSpace, SearchConfig,
    ServiceRequirements, search_design_space,
)

req = ServiceRequirements.degrees(
    min_elevation_deg=10, polar_latitude_deg=75,
    min_global_coverage=1.0, max_global_gap_s=2*3600,
    min_polar_coverage=1.0,  max_polar_gap_s=2*3600,
)
space = DesignSpace(
    altitudes_km=[1000, 1200], inclinations_deg=[53, 83],
    planes=[6, 7, 9], per_plane=[8, 10],   # all Walker F values tried
)
report = search_design_space(space, req, SearchConfig(
    duration_s=2*86400, step_s=240, grid_lat_step_deg=10))
```

The search (1) geometry-screens patterns whose ground track plus footprint
cannot reach the poles, (2) propagates every slot over an area-weighted grid
at several constellation phases, and (3) accepts/rejects criterion by
criterion — global ever/coverage, global worst service gap, polar coverage
and polar gap. A high mean coverage never produces a pass on its own; a
coarse pass that fails on a fine grid is rejected explicitly. Each accepted
plan carries the full per-satellite table (plane, slot, RAAN, mean anomaly,
semi-major axis, period).

Run the bundled study (search + fine-grid verification + plan tables):

```bash
PYTHONPATH=src python3 -m orbitforge.constellation.cli \
    --text-out plan.txt --json-out plan.json
```

HTTP endpoints: `POST /v1/constellation/evaluate` (one pattern) and
`POST /v1/constellation/search` (with optional fine-grid refinement).

