import torch
import numpy as np
import healpy as hp
from sklearn.neighbors import BallTree


def rad_to_xyz(lonlat: torch.Tensor):
    """Convert lon-lat (in radians) to unit sphere xyz."""
    lon = lonlat[..., 0]
    lat = lonlat[..., 1]

    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)

    return torch.stack([x, y, z], axis=-1)


def get_healpix_grid(nside: int) -> torch.Tensor:
    """Return HEALPix grid coordinates as array of shape (npix, 2)."""
    indices = np.arange(hp.nside2npix(nside))
    theta, phi = hp.pix2ang(nside, indices, nest=True)

    phi = np.rad2deg(phi)
    theta = (90. - np.rad2deg(theta))

    phi = torch.from_numpy(phi)
    theta = torch.from_numpy(theta)

    return torch.stack((phi, theta), axis=-1).float()


def get_neighbors(pos_from: np.ndarray, pos_to: np.ndarray, k: int = 8) -> tuple:
    """Build a BallTree and query k nearest neighbors with haversine metric."""
    pos_from_rad = pos_from[:, ::-1]
    pos_to_rad = pos_to[:, ::-1]

    tree = BallTree(pos_from_rad, metric='haversine')
    _, neighbors = tree.query(pos_to_rad, k=k)
    return neighbors
