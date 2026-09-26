"""Constellation pattern definition and instantiation into individual satellites.

A :class:`ConstellationPattern` describes a Walker delta constellation at the
orbital-element level (planes, in-plane slots, phasing, altitude, inclination).
:func:`instantiate` turns the pattern into one :class:`Satellite` record per
slot, carrying full Keplerian parameters for that satellite.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from orbitforge.core.constants import MU_EARTH_KM3_S2, R_EARTH_EQUATOR_KM
from orbitforge.coverage.constellation import walker_delta
from orbitforge.orbits.elements import KeplerianElements
from orbitforge.orbits.periods import mean_motion_rad_s, orbital_period_s

TWO_PI = 2.0 * math.pi
DEG = 180.0 / math.pi


def access_half_angle_rad(alt_km: float, min_elevation_rad: float = 0.0) -> float:
    """Geocentric half-angle of the radio/visibility footprint.

    Matches the geometry behind ``coverage.footprint.footprint_radius_km``:
    the satellite sees a ground point down to ``min_elevation_rad`` whenever
    the central angle between subpoint and target is at most this angle.
    """
    re = R_EARTH_EQUATOR_KM
    rs = re + alt_km
    return math.acos(max(-1.0, min(1.0, re / rs * math.cos(min_elevation_rad)))) - min_elevation_rad


def max_ground_latitude_rad(inclination_rad: float) -> float:
    """Highest geographic latitude the ground track can reach (radians)."""
    i = inclination_rad % TWO_PI
    return i if i <= math.pi / 2.0 else math.pi - i


@dataclass(frozen=True)
class ConstellationPattern:
    """Walker delta pattern T/P/F at a shared altitude and inclination."""

    satellites: int
    planes: int
    phasing: int
    alt_km: float
    inclination_rad: float
    eccentricity: float = 0.0
    argp_rad: float = 0.0
    raan_offset_rad: float = 0.0
    mean_anomaly_offset_rad: float = 0.0
    name: str = ""

    def __post_init__(self):
        if self.satellites <= 0 or self.planes <= 0:
            raise ValueError("satellites and planes must be positive")
        if self.satellites % self.planes != 0:
            raise ValueError("satellites must divide evenly among planes")
        per_plane = self.satellites // self.planes
        if not 0 <= self.phasing < per_plane:
            raise ValueError(f"walker phasing F must satisfy 0 <= F < {per_plane}")
        if self.alt_km <= 0:
            raise ValueError("altitude must be positive")
        if not 0.0 <= self.inclination_rad <= math.pi:
            raise ValueError("inclination must be in [0, pi]")
        if not 0.0 <= self.eccentricity < 1.0:
            raise ValueError("eccentricity must be in [0, 1)")

    @property
    def per_plane(self) -> int:
        return self.satellites // self.planes

    @property
    def semi_major_axis_km(self) -> float:
        return R_EARTH_EQUATOR_KM + self.alt_km

    @property
    def pattern_id(self) -> str:
        label = self.name or "Walker"
        return (
            f"{label}-{self.satellites}/{self.planes}/{self.phasing}"
            f"@{self.alt_km:g}km-{self.inclination_rad * DEG:.1f}deg"
        )


@dataclass(frozen=True)
class Satellite:
    """One instantiated satellite: its Walker slot plus orbital elements."""

    index: int
    name: str
    plane: int
    slot: int
    a_km: float
    e: float
    i_rad: float
    raan_rad: float
    argp_rad: float
    mean_anomaly_rad: float

    @property
    def alt_km(self) -> float:
        return self.a_km - R_EARTH_EQUATOR_KM

    @property
    def period_s(self) -> float:
        return orbital_period_s(self.a_km)

    def elements(self) -> KeplerianElements:
        # Mean anomaly stands in for true anomaly on a circular reference.
        return KeplerianElements(
            self.a_km,
            self.e,
            self.i_rad,
            self.raan_rad,
            self.argp_rad,
            self.mean_anomaly_rad,
        )


@dataclass
class ConstellationDesign:
    pattern: ConstellationPattern
    satellites: list[Satellite] = field(default_factory=list)
    epoch_s: float = 0.0

    @property
    def total(self) -> int:
        return len(self.satellites)

    @property
    def period_s(self) -> float:
        return orbital_period_s(self.pattern.semi_major_axis_km)

    @property
    def mean_motion_rad_s(self) -> float:
        return mean_motion_rad_s(self.pattern.semi_major_axis_km)


def instantiate(pattern: ConstellationPattern, epoch_s: float = 0.0, name_prefix: str = "S") -> ConstellationDesign:
    """Expand a Walker pattern into per-satellite orbital element records."""
    a = pattern.semi_major_axis_km
    slots = walker_delta(pattern.satellites, pattern.planes, pattern.phasing)
    sats = []
    for k, ws in enumerate(slots):
        raan = (ws.raan_rad + pattern.raan_offset_rad) % TWO_PI
        m0 = (ws.mean_anomaly_rad + pattern.mean_anomaly_offset_rad) % TWO_PI
        name = f"{name_prefix}{ws.plane + 1:02d}-{ws.slot + 1:02d}"
        sats.append(
            Satellite(
                index=k,
                name=name,
                plane=ws.plane,
                slot=ws.slot,
                a_km=a,
                e=pattern.eccentricity,
                i_rad=pattern.inclination_rad,
                raan_rad=raan,
                argp_rad=pattern.argp_rad,
                mean_anomaly_rad=m0,
            )
        )
    return ConstellationDesign(pattern=pattern, satellites=sats, epoch_s=epoch_s)
