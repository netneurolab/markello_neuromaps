# -*- coding: utf-8 -*-
"""
Functions for operating on images + surfaces
"""

import gzip
from pathlib import Path

import nibabel as nib
from nibabel.filebasedimages import ImageFileError
import numpy as np
from scipy.interpolate import griddata

from netneurotools.civet import read_civet


def construct_surf_gii(vert, tri):
    """
    Constructs surface gifti image from `vert` and `tri`

    Parameters
    ----------
    vert : (N, 3)
        Vertices of surface mesh
    tri : (T, 3)
        Triangles comprising surface mesh

    Returns
    -------
    img : nib.gifti.GiftiImage
        Surface image
    """

    vert = nib.gifti.GiftiDataArray(vert, 'NIFTI_INTENT_POINTSET',
                                    'NIFTI_TYPE_FLOAT32',
                                    coordsys=nib.gifti.GiftiCoordSystem(3, 3))
    tri = nib.gifti.GiftiDataArray(tri, 'NIFTI_INTENT_TRIANGLE',
                                   'NIFTI_TYPE_INT32')
    img = nib.GiftiImage(darrays=[vert, tri])

    return img


def construct_shape_gii(data, names=None, intent='NIFTI_INTENT_SHAPE'):
    """
    Constructs shape gifti image from `data`

    Parameters
    ----------
    data : (N[, F]) array_like
        Input data (where `F` corresponds to different features, if applicable)

    Returns
    -------
    img : nib.gifti.GiftiImage
        Shape image
    """

    if data.ndim == 1:
        data = data[:, None]
    if names is not None:
        if len(names) != data.shape[1]:
            raise ValueError('Length of provided `names` does not match '
                             'number of features in `data`')
        names = [{'Name': name} for name in names]
    else:
        names = [{} for _ in range(data.shape[1])]

    return nib.GiftiImage(darrays=[
        nib.gifti.GiftiDataArray(darr, intent=intent,
                                 datatype='NIFTI_TYPE_FLOAT32',
                                 meta=meta)
        for darr, meta in zip(data.T, names)
    ])


def fix_coordsys(fn, val=3):
    """
    Sets {xform,data}space of coordsys for GIFTI image `fn` to `val`

    Parameters
    ----------
    fn : str or os.PathLike
        Path to GIFTI image

    Returns
    -------
    fn : os.PathLike
        Path to GIFTI image
    """

    fn = Path(fn)
    img = nib.load(fn)
    for attr in ('dataspace', 'xformspace'):
        setattr(img.darrays[0].coordsys, attr, val)
    nib.save(img, fn)
    return fn


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


def obj_to_gifti(obj, fn=None):
    """
    Converts CIVET `obj` surface file to GIFTI format

    Parameters
    ----------
    obj : str or os.PathLike
        CIVET file to be converted
    fn : str or os.PathLike, None
        Output filename. If not supplied uses input `obj` filename (with
        appropriate suffix). Default: None

    Returns
    -------
    fn : os.PathLike
        Path to saved image file
    """

    img = construct_surf_gii(*read_civet(Path(obj)))
    if fn is None:
        fn = obj
    fn = Path(fn).resolve()
    if fn.name.endswith('.obj'):
        fn = fn.parent / fn.name.replace('.obj', '.surf.gii')
    nib.save(img, fn)

    return fn


def fssurf_to_gifti(surf, fn=None):
    """
    Converts FreeSurfer `surf` surface file to GIFTI format

    Parameters
    ----------
    obj : str or os.PathLike
        FreeSurfer surface file to be converted
    fn : str or os.PathLike, None
        Output filename. If not supplied uses input `surf` filename (with
        appropriate suffix). Default: None

    Returns
    -------
    fn : os.PathLike
        Path to saved image file
    """

    img = construct_surf_gii(*nib.freesurfer.read_geometry(Path(surf)))
    if fn is None:
        fn = surf + '.surf.gii'
    fn = Path(fn)
    nib.save(img, fn)

    return fn


def fsmorph_to_gifti(morph, fn=None, modifier=None):
    """
    Converts FreeSurfer `morph` data file to GIFTI format

    Parameters
    ----------
    obj : str or os.PathLike
        FreeSurfer morph file to be converted
    fn : str or os.PathLike, None
        Output filename. If not supplied uses input `morph` filename (with
        appropriate suffix). Default: None
    modifier : float, optional
        Scalar factor to modify (multiply) the morphometric data. Default: None

    Returns
    -------
    fn : os.PathLike
        Path to saved image file
    """

    data = nib.freesurfer.read_morph_data(Path(morph))
    if modifier is not None:
        data *= float(modifier)
    img = construct_shape_gii(data)
    if fn is None:
        fn = morph + '.shape.gii'
    fn = Path(fn)
    nib.save(img, fn)

    return fn


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

    src = load_gifti(src).agg_data('NIFTI_INTENT_POINTSET')
    data = load_gifti(data).agg_data()
    if len(src) != len(data):
        raise ValueError('Provided `src` file has different number of '
                         'vertices from `data` file')
    trg = load_gifti(trg).agg_data('NIFTI_INTENT_POINTSET')

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

    vert, tri = load_gifti(surface).agg_data()
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
        img = load_gifti(surf)
        vert = img.agg_data('NIFTI_INTENT_POINTSET')
        if vertices is None:
            vertices = np.zeros_like(vert)
        if triangles is None:
            triangles = img.agg_data('NIFTI_INTENT_TRIANGLE')
        vertices += vert

    vertices /= n_surfs

    return construct_surf_gii(vertices, triangles)
