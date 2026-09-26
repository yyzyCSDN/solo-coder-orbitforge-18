"""Keplerian propagation of constellation satellites to ground subpoints.

Circular Walker slots are propagated analytically (the hot path of the design
search). Eccentric slots fall back to the Kepler solver and a full ECI state
rotation. Earth rotation is included through the existing sidereal-time
utilities so ground-track drift and revisit are represented.
"""
from __future__ import annotations

import math

from orbitforge.core.constants import OMEGA_EARTH_RAD_S
from orbitforge.core.vector import Vec3
from orbitforge.frames.rotations import mv, rx, rz
from orbitforge.orbits.kepler import solve_kepler_elliptic, true_from_eccentric
from orbitforge.orbits.periods import mean_motion_rad_s

from .design import ConstellationDesign, Satellite, TWO_PI


def _mm3(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def eccentric_subpoint_rad(sat: Satellite, dt_s: float, gmst_rad: float):
    """Subpoint (lat, lon) of a possibly eccentric satellite via elements."""
    n = mean_motion_rad_s(sat.a_km)
    E = solve_kepler_elliptic((sat.mean_anomaly_rad + n * dt_s) % TWO_PI, sat.e)
    nu = true_from_eccentric(E, sat.e)
    r = sat.a_km * (1.0 - sat.e * math.cos(E))
    rp = Vec3(r * math.cos(nu), r * math.sin(nu), 0.0)
    q = _mm3(_mm3(rz(sat.raan_rad), rx(sat.i_rad)), rz(sat.argp_rad))
    pos = mv(q, rp)
    lat = math.asin(max(-1.0, min(1.0, pos.z / r)))
    lon = math.atan2(pos.y, pos.x) - gmst_rad
    return lat, lon


class ConstellationSampler:
    """Precomputed per-satellite constants for fast subpoint sampling."""

    def __init__(self, design: ConstellationDesign, phase_offset_s: float = 0.0):
        self.design = design
        self.base_epoch_s = design.epoch_s
        self.epoch_s = design.epoch_s + phase_offset_s
        self.n_rad_s = design.mean_motion_rad_s
        shift = self.n_rad_s * phase_offset_s
        self._circular = [s for s in design.satellites if s.e == 0.0]
        self._eccentric = [s for s in design.satellites if s.e != 0.0]
        self._const = [
            (
                math.sin(s.i_rad),
                math.cos(s.i_rad),
                math.sin(s.raan_rad),
                math.cos(s.raan_rad),
                (s.mean_anomaly_rad + shift) % TWO_PI,
            )
            for s in self._circular
        ]
        self._eccentric_m0 = [
            (s, (s.mean_anomaly_rad + shift) % TWO_PI) for s in self._eccentric
        ]
        # Greenwich angle is advanced linearly: gmst(epoch+dt) = gmst0 + w*dt,
        # accurate to well within one grid cell over the short search window.
        from orbitforge.time.sidereal import gmst_angle

        self._gmst0 = gmst_angle(self.epoch_s)
        self._omega = OMEGA_EARTH_RAD_S

    def sample(self, dt_s: float):
        """Return list of (lat_rad, lon_rad) subpoints, circular first."""
        gmst = self._gmst0 + self._omega * dt_s
        out = []
        u0 = self.n_rad_s * dt_s
        for si, ci, sr, cr, m0 in self._const:
            u = m0 + u0
            cu, su = math.cos(u), math.sin(u)
            x = cr * cu - sr * ci * su
            y = sr * cu + cr * ci * su
            z = si * su
            lat = math.asin(max(-1.0, min(1.0, z)))
            lon = math.atan2(y, x) - gmst
            out.append((lat, lon))
        for sat, m0 in self._eccentric_m0:
            shifted = Satellite(
                sat.index, sat.name, sat.plane, sat.slot, sat.a_km, sat.e,
                sat.i_rad, sat.raan_rad, sat.argp_rad, m0,
            )
            out.append(eccentric_subpoint_rad(shifted, dt_s, gmst))
        return out

    @property
    def mean_radius_km(self) -> float:
        return self.design.pattern.semi_major_axis_km
