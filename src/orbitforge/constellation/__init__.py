"""Constellation design: pattern instantiation, service evaluation, search."""
from orbitforge.constellation.design import (
    ConstellationDesign,
    ConstellationPattern,
    Satellite,
    access_half_angle_rad,
    instantiate,
    max_ground_latitude_rad,
)
from orbitforge.constellation.evaluator import (
    CellResult,
    GridCell,
    RegionMetrics,
    ServiceMetrics,
    area_weighted_grid,
    evaluate_pattern,
)
from orbitforge.constellation.propagation import ConstellationSampler
from orbitforge.constellation.requirements import (
    CriterionResult,
    ServiceRequirements,
    Verdict,
    geometric_prescreen,
    geometric_reach,
    verify,
)
from orbitforge.constellation.search import (
    CandidateResult,
    DesignSpace,
    SearchConfig,
    SearchReport,
    refine_candidate,
    search_design_space,
)

__all__ = [
    "ConstellationPattern",
    "ConstellationDesign",
    "Satellite",
    "access_half_angle_rad",
    "max_ground_latitude_rad",
    "instantiate",
    "ConstellationSampler",
    "GridCell",
    "CellResult",
    "RegionMetrics",
    "ServiceMetrics",
    "area_weighted_grid",
    "evaluate_pattern",
    "ServiceRequirements",
    "CriterionResult",
    "Verdict",
    "geometric_prescreen",
    "geometric_reach",
    "verify",
    "DesignSpace",
    "SearchConfig",
    "CandidateResult",
    "SearchReport",
    "search_design_space",
    "refine_candidate",
]
