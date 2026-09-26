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

`orbitforge.constellation.designer` turns orbital shells and Walker delta
patterns into concrete satellites, propagates them over a configurable horizon,
and scores each combination against *hard* service gates (not a mean coverage
figure):

- area-weighted global coverage over the whole horizon,
- area-weighted polar coverage above `|lat| >= polar_latitude_deg`,
- worst-case revisit / service-wait gap (edge waits included), with a p90 gap
  and the fraction of grid cells meeting the revisit limit.

```python
from orbitforge.constellation import (
    ServiceRequirements, design_search)

req = ServiceRequirements(min_global_coverage=0.98,
                          min_polar_coverage=0.90,
                          max_revisit_gap_s=2 * 3600)
report = design_search(
    'LEO',
    altitudes_km=[800, 1000, 1200],
    inclinations_deg=[53.0, 74.0, 83.0],
    total_satellites=[24, 36, 48, 60],
    plane_options=[6, 8, 12],
    phasing_options=[0, 1, 4],
    req=req)

# report.selected is the passing design with the fewest satellites, and each
# rejected candidate carries the exact gate(s) it failed.
for sat in report.satellites:          # slot + RAAN, M, a, e, i, period
    print(sat.sat_id, sat.raan_rad, sat.mean_anomaly_rad)
for cand in report.rejected:
    print(cand.failures)
```

HTTP: `POST /v1/constellation/design` runs the search (selected design, every
feasible and rejected candidate, per-satellite slots), and
`POST /v1/constellation/evaluate` scores a single Walker shell.
A worked report with the full per-satellite table is in
`examples/constellation_design_plan.md`.

