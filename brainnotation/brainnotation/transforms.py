# -*- coding: utf-8 -*-
"""
Functionality for transforming files between spaces
"""

import os
from pathlib import Path
from pkg_resources import resource_filename

import nibabel as nib
import numpy as np
from scipy.interpolate import interpn

from brainnotation.images import construct_shape_gii, load_gifti
from brainnotation.utils import tmpname
from netneurotools.utils import run

METRICRESAMPLE = 'wb_command -metric-resample {metric} {src} {trg} ' \
                 'ADAP_BARY_AREA {out} -area-metrics {srcarea} {trgarea} ' \
                 '-current-roi {srcmask}'
LABELRESAMPLE = 'wb_command -label-resample {label} {src} {trg} ' \
                'ADAP_BARY_AREA {out} -area-metrics {srcarea} {trgarea} ' \
                '-current-roi {srcmask}'
MASKSURF = 'wb_command -metric-mask {out} {trgmask} {out}'
ATLASDIR = Path(resource_filename('brainnotation', 'data/atlases')).resolve()
SURFFMT = 'tpl-{space}{trg}_den-{den}_hemi-{hemi}_sphere.surf.gii'
VAFMT = 'tpl-{space}_den-{den}_hemi-{hemi}_desc-vaavg_midthickness.shape.gii'
MLFMT = 'tpl-{space}_den-{den}_hemi-{hemi}_desc-nomedialwall_dparc.label.gii'
DENSITIES = dict(
    civet=['41k', '164k'],
    fsaverage=['3k', '10k', '41k', '164k'],
    fsLR=['4k', '8k', '32k', '164k'],
    MNI152=['1mm', '2mm', '3mm']
)


def _regfusion_project(data, ras, affine, method='linear'):
    """
    Project `data` to `ras` space using regfusion

    Parameters
    ----------
    data : (X, Y, Z[, V]) array_like
    ras : (N, 3) array_like
    affine (4, 4) array_like
    method : {'nearest', 'linear'}, optional

    Returns
    -------
    projected : (V, N) array_like
    """

    data, ras, affine = np.asarray(data), np.asarray(ras), np.asarray(affine)
    coords = nib.affines.apply_affine(np.linalg.inv(affine), ras)
    volgrid = [range(data.shape[i]) for i in range(3)]
    if data.ndim == 3:
        projected = interpn(volgrid, data, coords, method=method)
    elif data.ndim == 4:
        projected = np.column_stack([
            interpn(volgrid, data[..., n], coords, method=method)
            for n in range(data.shape[-1])
        ])

    return construct_shape_gii(projected.squeeze())


def _vol_to_surf(img, den, space, method='linear'):
    """
    Parameters
    ----------
    img : niimg_like, str, or os.PathLike
        Image to be projected to the surface
    den : str
        Density of desired output space
    space : str
        Desired output space
    method : {'nearest', 'linear'}, optional
        Method for projection. Default: 'linear'

    Returns
    -------
    projected : (2,) tuple-of-nib.GiftiImage
        Left [0] and right [1] hemisphere projected `image` data
    """

    if space not in DENSITIES:
        raise ValueError(f'Invalid space argument: {space}')
    if den not in DENSITIES[space]:
        raise ValueError(f'Invalid density for {space} space: {den}')
    if method not in ('nearest', 'linear'):
        raise ValueError('Invalid method argument: {method}')

    if isinstance(img, (str, os.PathLike)):
        img = nib.load(img)

    out = ()
    for hemi in 'LR':
        ras = ATLASDIR / 'regfusion' \
            / f'tpl-MNI152_space-{space}_den-{den}_hemi-{hemi}_regfusion.txt'
        out += (_regfusion_project(img.get_fdata(), np.loadtxt(ras),
                                   img.affine, method=method),)

    return out


def mni_to_civet(img, civet_density, method='linear'):
    return _vol_to_surf(img, civet_density, 'civet', method)


def mni_to_fsaverage(img, fsavg_density, method='linear'):
    return _vol_to_surf(img, fsavg_density, 'fsaverage', method)


def mni_to_fslr(img, fslr_density, method='linear'):
    return _vol_to_surf(img, fslr_density, 'fsLR', method)


def _check_hemi(data, hemi):
    """ Utility to check that `data` and `hemi` jive
    """
    if isinstance(data, (str, os.PathLike)) or not hasattr(data, '__len__'):
        data = (data,)
    if len(data) == 1 and hemi is None:
        raise ValueError('Must specify `hemi` when only 1 data file supplied')
    if hemi is not None and hemi not in ('L', 'R'):
        raise ValueError(f'Invalid hemisphere designation: {hemi}')
    elif hemi is not None:
        hemi = (hemi,)
    else:
        hemi = ('L', 'R')

    return zip(data, hemi)


def _surf_to_surf(data, srcparams, trgparams, method='linear', hemi=None):
    """
    Parameters
    ----------
    data : str or os.Pathlike or tuple
        Filepath(s) to data. If not a tuple then `hemi` must be specified. If
        a tuple then it is assumed that files are ('left', 'right')
    srcparams, trgparams : dict
        Dictionary with keys ['space', 'den', 'trg']
    method : {'nearest', 'linear'}, optional
        Method for projection. Default: 'linear'
    hemi : str or None
        Hemisphere of `data` if `data` is a single image. Default: None
    """

    methods = ('nearest', 'linear')
    if method not in methods:
        raise ValueError(f'Invalid method: {method}. Must be one of {methods}')

    keys = ('space', 'den', 'trg')
    for key in keys:
        if key not in srcparams:
            raise KeyError(f'srcparams missing key: {key}')
        if key not in trgparams:
            raise KeyError(f'trgparams missing key: {key}')

    for val in (srcparams, trgparams):
        space, den = val['space'], val['den']
        if den not in DENSITIES[space]:
            raise ValueError(f'Invalid density for {space} space: {den}')
    func = METRICRESAMPLE if method == 'linear' else LABELRESAMPLE

    out = ()
    for img, hemi in _check_hemi(data, hemi):
        srcparams['hemi'] = trgparams['hemi'] = hemi
        params = dict(
            metric=Path(img).resolve(),
            out=tmpname('.func.gii'),
            src=ATLASDIR / srcparams['space'] / SURFFMT.format(**srcparams),
            trg=ATLASDIR / trgparams['space'] / SURFFMT.format(**trgparams),
            srcarea=ATLASDIR / srcparams['space'] / VAFMT.format(**srcparams),
            trgarea=ATLASDIR / trgparams['space'] / VAFMT.format(**trgparams),
            srcmask=ATLASDIR / srcparams['space'] / MLFMT.format(**srcparams),
            trgmask=ATLASDIR / trgparams['space'] / MLFMT.format(**trgparams)
        )
        for fn in (func, MASKSURF):
            run(fn.format(**params), quiet=True)
        out += (construct_shape_gii(
            np.nan_to_num(load_gifti(params['out']).agg_data())
        ),)
        params['out'].unlink()

    return out


def civet_to_fslr(data, density, fslr_density='32k', hemi=None,
                  method='linear'):
    srcparams = dict(space='civet', den=density, trg='_space-fsLR')
    trgparams = dict(space='fsLR', den=fslr_density, trg='')
    return _surf_to_surf(data, srcparams, trgparams, method, hemi)


def fslr_to_civet(data, density, civet_density='41k', hemi=None,
                  method='linear'):
    srcparams = dict(space='fsLR', den=density, trg='')
    trgparams = dict(space='civet', den=civet_density, trg='_space-fsLR')
    return _surf_to_surf(data, srcparams, trgparams, method, hemi)


def civet_to_fsaverage(data, density, fsavg_density='41k', hemi=None,
                       method='linear'):
    srcparams = dict(space='civet', den=density, trg='_space-fsaverage')
    trgparams = dict(space='fsaverage', den=fsavg_density, trg='')
    return _surf_to_surf(data, srcparams, trgparams, method, hemi)


def fsaverage_to_civet(data, density, civet_density='41k', hemi=None,
                       method='linear'):
    srcparams = dict(space='fsaverage', den=density, trg='')
    trgparams = dict(space='civet', den=civet_density, trg='_space-fsaverage')
    return _surf_to_surf(data, srcparams, trgparams, method, hemi)


def fslr_to_fsaverage(data, density, fsavg_density='41k', hemi=None,
                      method='linear'):
    srcparams = dict(space='fsLR', den=density, trg='_space-fsaverage')
    trgparams = dict(space='fsaverage', den=fsavg_density, trg='')
    return _surf_to_surf(data, srcparams, trgparams, method, hemi)


def fsaverage_to_fslr(data, density, fslr_density='32k', hemi=None,
                      method='linear'):
    srcparams = dict(space='fsaverage', den=density, trg='')
    trgparams = dict(space='fsLR', den=fslr_density, trg='_space-fsaverage')
    return _surf_to_surf(data, srcparams, trgparams, method, hemi)


def civet_to_civet(data, density, civet_density='41k', hemi=None,
                   method='linear'):
    srcparams = dict(space='civet', den=density, trg='')
    trgparams = dict(space='civet', den=civet_density, trg='')
    return _surf_to_surf(data, srcparams, trgparams, method, hemi)


def fslr_to_fslr(data, density, fslr_density='41k', hemi=None,
                 method='linear'):
    srcparams = dict(space='fsLR', den=density, trg='')
    trgparams = dict(space='fsLR', den=fslr_density, trg='')
    return _surf_to_surf(data, srcparams, trgparams, method, hemi)


def fsaverage_to_fsaverage(data, density, fsavg_density='41k', hemi=None,
                           method='linear'):
    srcparams = dict(space='fsaverage', den=density, trg='')
    trgparams = dict(space='fsaverage', den=fsavg_density, trg='')
    return _surf_to_surf(data, srcparams, trgparams, method, hemi)
