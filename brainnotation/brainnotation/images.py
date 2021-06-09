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

PARCIGNORE = [
    'unknown', 'corpuscallosum', 'Background+FreeSurfer_Defined_Medial_Wall',
    '???'
]


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
        nib.gifti.GiftiDataArray(darr.astype('float32'), intent=intent,
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

    from brainnotation.civet import read_civet_surf

    img = construct_surf_gii(*read_civet_surf(Path(obj)))
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

    Parameters
    ----------
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


def _relabel(labels, minval=0, bgval=None):
    """
    Relabels `labels` so that they're consecutive

    Parameters
    ----------
    labels : (N,) array_like
        Labels to be re-labelled
    minval : int, optional
        What the new minimum value of the labels should be. Default: 0
    bgval : int, optional
        What the background value should be; the new labels will start at
        `minval` but the first value of these labels (i.e., labels == `minval`)
        will be set to `bgval`. Default: None

    Returns
    ------
    labels : (N,) np.ndarray
        New labels
    """

    labels = np.unique(labels, return_inverse=True)[-1] + minval
    if bgval is not None:
        labels[labels == minval] = bgval
    return labels


def relabel_gifti(parcellation, background=None, offset=None):
    """
    Updates GIFTI images so label IDs are consecutive across hemispheres

    Parameters
    ----------
    parcellation : (2,) tuple-of-str
        Surface label files in GIFTI format (lh.label.gii, rh.label.gii)
    background : list-of-str, optional
        If provided, a list of IDs in `parcellation` that should be set to 0
        (the presumptive background value). Other IDs will be shifted so they
        are consecutive (i.e., 0--N). If not specified will use labels in
        `brainnotation.images.PARCIGNORE`. Default: None
    offset : int, optional
        What the lowest value in `parcellation[1]` should be not including
        background value. If not specified it will be purely consecutive from
        `parcellation[0]`. Default: None

    Returns
    -------
    relabelled : (2,) tuple-of-nib.gifti.GiftiImage
        Re-labelled `parcellation` files
    """

    relabelled = tuple()
    minval = 0
    if not isinstance(parcellation, tuple):
        parcellation = (parcellation,)

    if background is None:
        background = PARCIGNORE.copy()

    for hemi in parcellation:
        # get necessary info from file
        img = load_gifti(hemi)
        data = img.agg_data()
        labels = img.labeltable.labels
        lt = {v: k for k, v in img.labeltable.get_labels_as_dict().items()}

        # get rid of labels we want to drop
        if background is not None:
            for val in background:
                idx = lt.get(val, 0)
                if idx == 0:
                    continue
                data[data == idx] = 0
                labels = [f for f in labels if f.key != idx]

        # reset labels so they're consecutive and update label keys
        data = _relabel(data, minval=minval, bgval=0)
        ids = np.unique(data)
        new_labels = []
        for n, i in enumerate(ids):
            lab = labels[n]
            lab.key = i
            new_labels.append(lab)
        minval = len(ids) - 1 if offset is None else int(offset) - 1

        # make new gifti image with updated information
        darr = nib.gifti.GiftiDataArray(data, intent='NIFTI_INTENT_LABEL',
                                        datatype='NIFTI_TYPE_INT32')
        labeltable = nib.gifti.GiftiLabelTable()
        labeltable.labels = new_labels
        img = nib.GiftiImage(darrays=[darr], labeltable=labeltable)
        relabelled += (img,)

    return relabelled


def annot_to_gifti(parcellation):
    """
    Converts FreeSurfer-style annotation `parcellation` files to GIFTI images

    Parameters
    ----------
    parcellation : tuple of str or os.PathLike
        Paths to surface annotation files (.annot)

    Returns
    -------
    gifti : tuple-of-nib.GiftiImage
        Converted GIFTI images
    """

    if not isinstance(parcellation, tuple):
        parcellation = (parcellation,)

    gifti = tuple()
    for atlas in parcellation:
        labels, ctab, names = nib.freesurfer.read_annot(atlas)

        darr = nib.gifti.GiftiDataArray(labels, intent='NIFTI_INTENT_LABEL',
                                        datatype='NIFTI_TYPE_INT32')
        labeltable = nib.gifti.GiftiLabelTable()
        for key, label in enumerate(names):
            (r, g, b), a = (ctab[key, :3] / 255), (1.0 if key != 0 else 0.0)
            glabel = nib.gifti.GiftiLabel(key, r, g, b, a)
            glabel.label = label.decode()
            labeltable.labels.append(glabel)

        gifti += (nib.GiftiImage(darrays=[darr], labeltable=labeltable),)

    return gifti


def dlabel_to_gifti(parcellation):
    """
    Converts CIFTI dlabel file to GIFTI images

    Parameters
    ----------
    parcellation : str or os.PathLike
        Path to CIFTI parcellation file (.dlabel.nii)

    Returns
    -------
    gifti : tuple-of-nib.GiftiImage
        Converted GIFTI images
    """

    structures = ('CORTEX_LEFT', 'CORTEX_RIGHT')

    dlabel = nib.load(parcellation)
    parcdata = np.asarray(dlabel.get_fdata(), dtype='int32').squeeze()

    gifti = tuple()
    label_dict = dlabel.header.get_axis(index=0).label[0]
    for bm in dlabel.header.get_index_map(1).brain_models:
        structure = bm.brain_structure
        if structure.startswith('CIFTI_STRUCTURE_'):
            structure = structure[16:]
        if structure not in structures:
            continue
        labels = np.zeros(bm.surface_number_of_vertices, dtype='int32')
        idx = np.asarray(bm.vertex_indices)
        slicer = slice(bm.index_offset, bm.index_offset + bm.index_count)
        labels[idx] = parcdata[slicer]

        darr = nib.gifti.GiftiDataArray(labels, intent='NIFTI_INTENT_LABEL',
                                        datatype='NIFTI_TYPE_INT32')
        labeltable = nib.gifti.GiftiLabelTable()
        for key, (label, (r, g, b, a)) in label_dict.items():
            if key not in labels:
                continue
            glabel = nib.gifti.GiftiLabel(key, r, g, b, a)
            glabel.label = label
            labeltable.labels.append(glabel)
        gifti += (nib.GiftiImage(darrays=[darr], labeltable=labeltable),)

    return gifti
