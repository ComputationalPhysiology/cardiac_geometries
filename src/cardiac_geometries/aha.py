import dolfin
import numpy as np

from .dolfin_utils import Geometry


def focal(r_long_endo: float, r_short_endo: float):
    return np.sqrt(r_long_endo**2 - r_short_endo**2)


def full_arctangent(x, y):
    t = np.arctan2(x, y)
    if t < 0:
        return t + 2 * np.pi
    else:
        return t


def cartesian_to_prolate_ellipsoidal(x, y, z, a):
    b1 = np.sqrt((x + a) ** 2 + y**2 + z**2)
    b2 = np.sqrt((x - a) ** 2 + y**2 + z**2)

    sigma = 1 / (2.0 * a) * (b1 + b2)
    tau = 1 / (2.0 * a) * (b1 - b2)
    phi = full_arctangent(z, y)
    nu = np.arccosh(sigma)
    mu = np.arccos(tau)
    return nu, mu, phi


def get_level(regions, mu):
    A = np.intersect1d(
        np.where((regions.T[3] <= mu))[0],
        np.where((mu <= regions.T[0]))[0],
    )
    if len(A) == 0:
        return [np.shape(regions)[0] + 1]
    else:
        return A


def get_sector(regions, theta):
    if not (np.count_nonzero(regions.T[1] <= regions.T[2]) >= 0.5 * np.shape(regions)[0]):
        raise ValueError("Surfaces are flipped")

    sectors = []
    for i, r in enumerate(regions):
        if r[1] == r[2]:
            sectors.append(i)
        else:
            if r[1] > r[2]:
                if theta > r[1] or r[2] > theta:
                    sectors.append(i)

            else:
                if r[1] < theta < r[2]:
                    sectors.append(i)

    return sectors


class AHA_LV_17(dolfin.UserExpression):
    def __init__(self, foc: float, mu_base: float) -> None:
        super().__init__()
        self.mu_base = abs(mu_base)
        self.foc = foc

    @property
    def dmu(self) -> float:
        return (np.pi - self.mu_base) / 4

    def eval_cell(self, value, x, ufc_cell):
        nu, mu, phi = cartesian_to_prolate_ellipsoidal(*x, a=self.foc)

        if self.mu_base < mu <= self.mu_base + self.dmu:
            # BASE
            if 0 < phi <= np.pi / 3:
                value[0] = 1
            elif np.pi / 3 < phi <= 2 * np.pi / 3:
                value[0] = 2
            elif 2 * np.pi / 3 < phi <= np.pi:
                value[0] = 3
            elif np.pi < phi <= 4 * np.pi / 3:
                value[0] = 4
            elif 4 * np.pi / 3 < phi <= 5 * np.pi / 3:
                value[0] = 5
            else:
                value[0] = 6

        elif self.mu_base + self.dmu < mu <= self.mu_base + 2 * self.dmu:
            # MID
            if 0 < phi <= np.pi / 3:
                value[0] = 7
            elif np.pi / 3 < phi <= 2 * np.pi / 3:
                value[0] = 8
            elif 2 * np.pi / 3 < phi <= np.pi:
                value[0] = 9
            elif np.pi < phi <= 4 * np.pi / 3:
                value[0] = 10
            elif 4 * np.pi / 3 < phi <= 5 * np.pi / 3:
                value[0] = 11
            else:
                value[0] = 12
        elif self.mu_base + 2 * self.dmu < mu <= self.mu_base + 3 * self.dmu:
            # APICAL
            if 0 < phi <= np.pi / 2:
                value[0] = 13
            elif np.pi / 2 < phi <= np.pi:
                value[0] = 14
            elif np.pi < phi <= 3 * np.pi / 2:
                value[0] = 15
            else:
                value[0] = 16
        else:
            value[0] = 17

    def value_shape(self):
        return ()


def lv_aha(
    geometry: Geometry,
    r_long_endo: float,
    r_short_endo: float,
    mu_base: float,
) -> Geometry:
    foc = focal(r_long_endo=r_long_endo, r_short_endo=r_short_endo)

    V = dolfin.FunctionSpace(geometry.mesh, "DG", 0)
    # f = dolfin.Function(V)
    expr = AHA_LV_17(foc=foc, mu_base=mu_base)
    f = dolfin.interpolate(expr, V)
    geometry.marker_functions.cfun.array()[:] = f.vector().get_local()
    i = 1

    for level in ["BASAL", "MID", "APICAL"]:
        for sector in [
            "ANTERIOR",
            "ANTEROSEPTAL",
            "SEPTAL",
            "INFERIOR",
            "POSTERIOR",
            "LATERAL",
        ]:
            if level == "APICAL" and sector in ("ANTEROSEPTAL", "POSTERIOR"):
                continue

            geometry.markers["-".join((level, sector))] = (i, 3)
            i += 1
    geometry.markers["APEX"] = (17, 3)
    geometry.markers.pop("MYOCARDIUM")
    return geometry


def biv_aha(): ...
