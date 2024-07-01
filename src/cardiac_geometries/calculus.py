import numpy as np


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
    mu = np.arccosh(sigma)
    nu = np.arccos(tau)
    return mu, nu, phi


def prolate_ellipsoidal_to_cartesian(mu, nu, phi, a):
    x = a * np.cosh(mu) * np.cos(nu)
    y = a * np.sinh(mu) * np.sin(nu) * np.cos(phi)
    z = a * np.sinh(mu) * np.sin(nu) * np.sin(phi)
    return x, y, z


def get_strain_regions(nsectors=(6, 6, 4, 1)):
    """Mark the cells in the mesh.

    For instance if you want to mark this mesh accoring to
    the  AHA 17-segment model, then nsector = [6,6,4,1],
    i.e 6 basal, 6 mid, 4 apical and one apex

    """

    nlevels = len(nsectors)

    pi = np.pi

    assert nlevels <= 4

    if nlevels == 4:
        mus = [90, 60, 30, 10, 0]
    elif nlevels == 3:
        mus = [90, 60, 30, 0]
    elif nlevels == 2:
        mus = [90, 45, 0]
    else:
        mus = [90, 0]

    thetas = [np.linspace(pi, 3 * pi, s + 1)[:-1].tolist() + [pi] for s in nsectors]

    start = 0
    end = nsectors[0]
    regions = np.zeros((sum(nsectors), 4))
    for i in range(nlevels):
        regions.T[0][start:end] = mus[i] * pi / 180
        regions.T[3][start:end] = mus[i + 1] * pi / 180
        if i != len(nsectors) - 1:
            start += nsectors[i]
            end += nsectors[i + 1]

    start = 0
    for j, t in enumerate(thetas):
        for i in range(nsectors[j]):
            regions.T[1][i + start] = t[i]
            regions.T[2][i + start] = t[i + 1]
        start += nsectors[j]

    return regions


def strain_region_number(T, regions):
    """
    For a given point in prolate coordinates,
    return the region it belongs to.

    :param regions: Array of all coordinates for the strain
                    regions taken from the strain mesh.
    :type regions: :py:class:`numpy.array`

    :param T: Some value i prolate coordinates
    :type T: :py:class:`numpy.array`

    Resturn the region number that
    T belongs to
    """

    """
    The cricumferential direction is a bit
    tricky because it goes from -pi to pi.
    To overcome this we add pi so that the
    direction goes from 0 to 2*pi
    """

    lam, mu, theta = T

    theta = theta + np.pi

    levels = get_level(regions, mu)

    if np.shape(regions)[0] + 1 in levels:
        return np.shape(regions)[0] + 1

    sector = get_sector(regions, theta)

    assert len(np.intersect1d(levels, sector)) == 1

    return np.intersect1d(levels, sector)[0] + 1


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


def estimate_focal_point(mesh, axis=0):
    r"""Copmute the focal point based on approximating the
    endocardial surfaces as a ellipsoidal cap.

    .. math::

           focal = \sqrt{ l^2 - s^2}


    Arguments
    ---------
    mesh: `dolfin.mesh`
        The mesh

    Returns
    -------
    focal_point: float
        The focal point

    """

    max_coord = np.max(mesh.coordinates(), axis)
    min_coord = np.min(mesh.coordinates(), axis)

    axis = np.abs(max_coord - min_coord)
    long_axis = np.max(axis)
    short_axis = np.min(axis)

    focal = np.sqrt(long_axis**2 - short_axis**2)

    return focal
