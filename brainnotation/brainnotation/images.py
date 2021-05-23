# -*- coding: utf-8 -*-
"""
Functions for operating on images + surfaces
"""

import gzip

import nibabel as nib
from nibabel.filebasedimages import ImageFileError
import numpy as np
from scipy.interpolate import griddata

from brainnotation.civet import construct_surf_gii


def interp_surface(data, src, trg, method='nearest'):
    """
    Interpolate `data` on `src` surface to `trg` surface

    Parameters
    ----------
    data : str or os.PathLike
        Path to (gifti) data file defined on `src` surface
    src : str or os.PathLike
        Path to (gifti) file defining surface of `data`
    trg : str or os.PathLike
        Path to (gifti) file defining desired output surface
    method : {'nearest', 'linear'}
        Method for interpolation. Default {'nearest'}

    Returns
    -------
    interp : np.ndarray
        Input `data` interpolated to `trg` surface
    """

    if method not in ('nearest', 'linear'):
        raise ValueError(f'Provided method {method} invalid')

    src = nib.load(src).agg_data('NIFTI_INTENT_POINTSET')
    data = nib.load(data).agg_data()
    if len(src) != len(data):
        raise ValueError('Provided `src` file has different number of '
                         'vertices from `data` file')
    trg = nib.load(trg).agg_data('NIFTI_INTENT_POINTSET')

    return griddata(src, data, trg, method=method)


def vertex_areas(surface):
    """
    Calculates vertex areas from `surface` file

    Vertex area is calculated as the sum of 1/3 the area of each triangle in
    which the vertex participates

    Parameter
    ---------
    surface : str or os.PathLike
        Path to (gifti) file defining surface for which areas should be
        computed

    Returns
    -------
    areas : np.ndarray
        Vertex areas
    """

    vert, tri = nib.load(surface).agg_data()
    vectors = np.diff(vert[tri], axis=1)
    cross = np.cross(vectors[:, 0], vectors[:, 1])
    triareas = (np.sqrt(np.sum(cross ** 2, axis=1)) * 0.5) / 3
    areas = np.bincount(tri.flatten(), weights=np.repeat(triareas, 3))

    return areas


def average_surfaces(*surfs):
    """
    Generates average surface from input `surfs`

    Parameters
    ----------
    surfs : str or os.PathLike
        Path to (gifti) surfaces to be averaged. Surfaces should be aligned!

    Returns
    -------
    average : nib.gifti.GiftiImage
        Averaged surface
    """

    n_surfs = len(surfs)
    vertices = triangles = None
    for surf in surfs:
        img = nib.load(surf)
        vert = img.agg_data('NIFTI_INTENT_POINTSET')
        if vertices is None:
            vertices = np.zeros_like(vert)
        if triangles is None:
            triangles = img.agg_data('NIFTI_INTENT_TRIANGLES')
        vertices += vert

    vertices /= n_surfs

    return construct_surf_gii(vertices, triangles)


def load_gifti(img):
    """
    Loads gifti file `img`

    Will try to gunzip `img` if gzip is detected, and will pass pre-loaded
    GiftiImage object

    Parameters
    ----------
    img : os.PathLike or nib.GiftiImage object
        Image to be loaded

    Returns
    -------
    img : nib.GiftiImage
        Loaded GIFTI images
    """

    try:
        img = nib.load(img)
    except (ImageFileError, TypeError) as err:
        # it's gzipped, so read the gzip and pipe it in
        if isinstance(err, ImageFileError) and str(err).endswith('.gii.gz"'):
            with gzip.GzipFile(img) as gz:
                img = nib.GiftiImage.from_bytes(gz.read())
        # it's not a pre-loaded GiftiImage so error out
        elif (isinstance(err, TypeError)
              and not str(err) == 'stat: path should be string, bytes, os.'
                                  'PathLike or integer, not GiftiImage'):
            raise err

    return img
