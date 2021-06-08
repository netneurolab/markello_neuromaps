# -*- coding: utf-8 -*-
"""
Functions for comparing data
"""

from typing import Iterable

import nibabel as nib
import numpy as np
from scipy.stats import rankdata

from brainnotation import transforms
from brainnotation.datasets import ALIAS, DENSITIES
from brainnotation.images import load_gifti


_resampling_docs = dict(
    resample_in="""\
{src, trg} : str or os.PathLike or niimg_like or nib.GiftiImage or tuple
    Input data to be resampled to each other
{src,trg}_space : str
    Template space of {`src`, `trg`} data\
hemi : {'L', 'R'}, optional
    If `src` and `trg` are not tuples this specifies the hemisphere the data
    represent. Default: None
method : {'nearest', 'linear'}, optional
    Method for resampling. Specify 'nearest' if `data` are label images.
    Default: 'linear'\
""",
    resample_out="""\
src, trg : tuple-of-nib.GiftiImage
    Resampled images\
"""
)


def _load_data(data):
    """ Small utility to load + stack `data` images (gifti / nifti)
    """

    out = ()
    for img in data:
        try:
            out += (load_gifti(img).agg_data(),)
        except (AttributeError, TypeError):
            if isinstance(img, str):
                img = nib.load(img)
            out += (img.get_fdata(),)
    return np.hstack(out)


def imgcorr(src, trg, corrtype='pearson', ignore_zero=True):
    """
    Correlates images `src` and `trg`

    If `src` and `trg` represent data from multiple hemispheres the hemispheres
    are concatenated prior to correlation

    Parameters
    ----------
    src, trg : str or os.PathLike or nib.GiftiImage or tuple
        Images to be correlated
    corrtype : {'pearson', 'spearman'}, optional
        Type of correlation to perform. Default: 'pearson'
    ignore_zero : bool, optional
        Whether to perform correlations ignoring all zero values in `src` and
        `trg` data. Default: True

    Returns
    -------
    correlation : float
         Correlation between `src` and `trg`
    """

    methods = ('pearson', 'spearman')
    if corrtype not in methods:
        raise ValueError(f'Invalid method: {corrtype}')

    if isinstance(src, str) or not isinstance(src, Iterable):
        src = (src,)
    if isinstance(trg, str) or not isinstance(trg, Iterable):
        trg = (trg,)

    srcdata = _load_data(src)
    trgdata = _load_data(trg)

    if ignore_zero:
        mask = np.logical_or(np.isclose(srcdata, 0), np.isclose(trgdata, 0))
        srcdata, trgdata = srcdata[~mask], trgdata[~mask]

    # drop NaNs
    mask = np.logical_or(np.isnan(srcdata), np.isnan(trgdata))
    srcdata, trgdata = srcdata[~mask], trgdata[~mask]

    if corrtype == 'spearman':
        srcdata, trgdata = rankdata(srcdata), trgdata(rankdata)

    return np.corrcoef(srcdata, trgdata)[0, 1]


def _estimate_density(data, hemi=None):
    """
    Tries to estimate standard density of `data`

    Parameters
    ----------
    data : (2,) tuple of str or os.PathLike or nib.GiftiImage or tuple
        Input data for (src, trg)
    hemi : {'L', 'R'}, optional
        If `data` is not a tuple this specifies the hemisphere the data are
        representing. Default: None

    Returns
    -------
    density : str
        String representing approximate density of data (e.g., '10k')

    Raises
    ------
    ValueError
        If density of `data` is not one of the standard expected values
    """

    density_map = {
        2562: '3k', 4002: '4k', 7842: '8k', 10242: '10k',
        32492: '32k', 40692: '41k', 163842: '164k'
    }

    densities = tuple()
    for img in data:
        img, hemi = zip(*transforms._check_hemi(img, hemi))
        n_vert = [len(load_gifti(d).agg_data()) for d in img]
        if not all(n == n_vert[0] for n in n_vert):
            raise ValueError('Provided data have different resolutions across '
                             'hemispheres?')
        else:
            n_vert = n_vert[0]
        density = density_map.get(n_vert)
        if density is None:
            raise ValueError('Provided data resolution is non-standard. '
                             'Number of vertices estimated in data: {n_vert}')
        densities += (density,)

    return densities


def downsample_only(src, trg, src_space, trg_space, hemi=None,
                    method='linear'):
    src_den, trg_den = _estimate_density((src, trg), hemi)
    src_num, trg_num = int(src_den[:-1]), int(trg_den[:-1])
    src_space, trg_space = src_space.lower(), trg_space.lower()

    if src_num >= trg_num:  # resample to `trg`
        func = getattr(transforms, f'{src_space}_to_{trg_space}')
        src = func(src, src_den, trg_den, hemi=hemi, method=method)
    elif src_num < trg_num:  # resample to `src`
        func = getattr(transforms, f'{trg_space}_to_{src_space}')
        trg = func(trg, trg_den, src_den, hemi=hemi, method=method)

    return src, trg


downsample_only.__doc__ = """\
Resamples `src` and `trg` to match such that neither is upsampled

If density of `src` is greater than `trg` then `src` is resampled to
`trg`; otherwise, `trg` is resampled to `src`

Parameters
----------
{resample_in}

Returns
-------
{resample_out}
""".format(**_resampling_docs)


def transform_to_src(src, trg, src_space, trg_space, hemi=None,
                     method='linear'):
    src_den, trg_den = _estimate_density((src, trg), hemi)

    func = getattr(transforms, f'{trg_space.lower()}_to_{src_space.lower()}')
    trg = func(trg, trg_den, src_den, hemi=hemi, method=method)

    return src, trg


transform_to_src.__doc__ = """\
Resamples `trg` to match space and density of `src`

Parameters
----------
{resample_in}

Returns
-------
{resample_out}
""".format(**_resampling_docs)


def transform_to_trg(src, trg, src_space, trg_space, hemi=None,
                     method='linear'):
    src_den, trg_den = _estimate_density((src, trg), hemi)

    func = getattr(transforms, f'{src_space.lower()}_to_{trg_space.lower()}')
    src = func(src, src_den, trg_den, hemi=hemi, method=method)

    return src, trg


transform_to_trg.__doc__ = """\
Resamples `trg` to match space and density of `src`

Parameters
----------
{resample_in}

Returns
-------
{resample_out}
""".format(**_resampling_docs)


def transform_to_alt(src, trg, src_space, trg_space, hemi=None,
                     method='linear', alt_space='fsaverage',
                     alt_density='41k'):
    src_den, trg_den = _estimate_density((src, trg), hemi)

    func = getattr(transforms, f'{src_space.lower()}_to_{alt_space.lower()}')
    src = func(src, src_den, alt_density, hemi=hemi, method=method)

    func = getattr(transforms, f'{trg_space.lower()}_to_{alt_space.lower()}')
    trg = func(trg, trg_den, alt_density, hemi=hemi, method=method)

    return src, trg


transform_to_alt.__doc__ = """\
Resamples `src` and `trg` to `alt_space` and `alt_density`

Parameters
----------
{resample_in}

Returns
-------
{resample_out}
""".format(**_resampling_docs)


def mni_transformation(src, trg, src_space, trg_space):
    if src_space != 'MNI152':
        raise ValueError('Cannot perform MNI transformation when src_space is '
                         f'not "MNI152." Received: {src_space}.')
    trg_den = trg
    if trg_space != 'MNI152':
        trg_den, = _estimate_density((trg_den,), None)
    func = getattr(transforms, f'mni152_to_{trg_space.lower()}')
    src = func(src, trg_den)
    return src, trg


def _check_altspec(spec):
    """
    Confirms that specified alternative `spec` is valid (space, density) format

    Parameters
    ----------
    spec : (2,) tuple-of-str
        Where entries are (space, density) of desired target space

    Returns
    -------
    spec : (2,) tuple-of-str
        Unmodified input `spec`

    Raises
    ------
    ValueError
        If `spec` is not valid format
    """

    invalid_spec = spec is None or len(spec) != 2
    if not invalid_spec:
        space, den = spec
        valid = DENSITIES.get(space)
        invalid_spec = valid is None or den not in valid
    if invalid_spec:
        raise ValueError('Must provide valid alternative specification of '
                         f'format (space, density). Received: {spec}')

    return (ALIAS.get(spec[0], spec[0]), spec[1])


def correlate_images(src, trg, src_space, trg_space, hemi=None,
                     method='linear', resampling='downsample_only',
                     corrtype='pearson', ignore_zero=True,
                     alt_spec=None):

    resamplings = ('downsample_only', 'transform_to_src', 'transform_to_trg',
                   'transform_to_alt')
    if resampling not in resamplings:
        raise ValueError(f'Invalid method: {resampling}')

    src_space = ALIAS.get(src_space, src_space)
    trg_space = ALIAS.get(trg_space, trg_space)

    opts, err = {}, None
    if resampling == 'transform_to_alt':
        opts['alt_space'], opts['alt_density'] = _check_altspec(alt_spec)
    elif (resampling == 'transform_to_src' and src_space == 'MNI152'
            and trg_space != 'MNI152'):
        err = ('Specified `src_space` cannot be "MNI152" when `resampling` is '
               '"transform_to_src"')
    elif (resampling == 'transform_to_trg' and src_space != 'MNI152'
            and trg_space == 'MNI152'):
        err = ('Specified `trg_space` cannot be "MNI152" when `resampling` is '
               '"transform_to_trg"')
    elif (resampling == 'transform_to_alt' and opts['alt_space'] == 'MNI152'
            and (src_space != 'MNI152' or trg_space != 'MNI152')):
        err = ('Specified `alt_space` cannot be "MNI152" when `resampling` is '
               '"transform_to_alt"')
    if err is not None:
        raise ValueError(err)

    if src_space == 'MNI152':
        src, trg = mni_transformation(src, trg, src_space, trg_space)
    elif trg_space == 'MNI152':
        trg, src = mni_transformation(trg, src, trg_space, src_space)
    else:
        func = globals()[resampling]
        src, trg = func(src, trg, src_space, trg_space, hemi=hemi,
                        method=method, **opts)
    correlation = imgcorr(src, trg, corrtype=corrtype, ignore_zero=ignore_zero)

    return correlation


correlate_images.__doc__ = """\
Correlates images `src` and `trg`, resampling as needed

Parameters
----------
{resample_in}
resampling : str, optional
    Name of resampling function to resample `src` and `trg`. Must be one of:
    {'downsample_only', 'transform_to_src', 'transform_to_trg',
    'transform_to_alt'}. See Notes for more info. Default: 'downsample_only'
corrtype : {'pearson', 'spearman'}, optional
    Type of correlation to perform. Default: 'pearson'
ignore_zero : bool, optional
    Whether to perform correlations ignoring all zero values in `src` and
    `trg` data. Default: True
alt_spec : (2,) tuple-of-str
    Where entries are (space, density) of desired target space. Only used if
    `resampling='transform_to_alt'. Default: None

Returns
-------
correlation : float
    Correlation between `src` and `trg`. If multiple hemispheres are
    provided then correlation is calculated between concatenated data

Notes
-----
The four available `resampling` strategies will control how `src` and/or `trg`
are resampled prior to correlation.
"""
