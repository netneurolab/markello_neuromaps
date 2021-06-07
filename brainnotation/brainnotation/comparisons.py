# -*- coding: utf-8 -*-
"""
Functions for comparing data
"""

from typing import Iterable

import numpy as np
from scipy.stats import rankdata

from brainnotation import transforms
from brainnotation.datasets import DENSITIES
from brainnotation.images import load_gifti


_resampling_docs = dict(
    resample_in="""\
{src, trg} : str or os.PathLike or nib.GiftiImage or tuple
    Input data to be resampled to each other
{src,trg}_space : str
    Template space of {`src`, `trg`} data\
hemi : {'L', 'R'}, optional
    If `src` and `trg` are not tuples this specifies the hemisphere the data
    represent. Default: None
method : {'nearest', 'linear'}, optional
    Method for resampling. Specify 'nearest' if `data` are label images.
    Default: 'linear'
""",
    resample_out="""\
src, trg : tuple-of-nib.GiftiImage
    Resampled images\
"""
)


def imgcorr(src, trg, corrtype='pearson', ignore_zero=True):
    """
    Correlates images `src` and `trg`

    Parameters
    ----------
    src, trg : str or os.PathLike or nib.GiftiImage
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

    if isinstance(src, str) or not isinstance(trg, Iterable):
        src = (src,)
    if isinstance(trg, str) or not isinstance(trg, Iterable):
        trg = (trg,)

    srcdata = np.hstack([np.nan_to_num(load_gifti(s).agg_data()) for s in src])
    trgdata = np.hstack([np.nan_to_num(load_gifti(t).agg_data()) for t in trg])
    if ignore_zero:
        mask = np.logical_and(np.isclose(srcdata, 0), np.isclose(trgdata, 0))
        srcdata, trgdata = srcdata[~mask], trgdata[~mask]
    if corrtype == 'spearman':
        srcdata, trgdata = rankdata(srcdata), trgdata(rankdata)

    return np.corrcoef(srcdata, trgdata)[0, 1]


def _estimate_density(data, hemi=None):
    """
    Tries to estimate standard density of `data`

    Parameters
    ----------
    data : str or os.PathLike or nib.GiftiImage or tuple
        Input data
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

    data, hemi = zip(*transforms._check_hemi(data, hemi))
    n_vert = [len(load_gifti(d).agg_data()) for d in data]
    if not all(n == n_vert[0] for n in n_vert):
        raise ValueError('Provided data have different resolutions across '
                         'hemispheres?')
    else:
        n_vert = n_vert[0]
    density = density_map.get(n_vert)
    if density is None:
        raise ValueError('Provided data resolution is non-standard. Number of '
                         f'vertices estimated in data: {n_vert}')

    return density


def downsample_only(src, trg, src_space, trg_space, hemi=None,
                    method='linear'):
    densities = {'src': None, 'trg': None}
    for data, key in zip((src, trg), densities):
        densities[key] = _estimate_density(data, hemi)

    src_den, trg_den = densities['src'], densities['trg']
    src_num, trg_num = int(src_den[:-1]), int(trg_den[:-1])

    if src_num >= trg_num:  # resample to `trg`
        func = getattr(transforms,
                       f'{src_space.lower()}_to_{trg_space.lower()}')
        src = func(src, src_den, trg_den, hemi=hemi, method=method)
    elif src_num < trg_num:  # resample to `src`
        func = getattr(transforms,
                       f'{trg_space.lower()}_to_{src_space.lower()}')
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
    densities = {'src': None, 'trg': None}
    for data, key in zip((src, trg), densities):
        densities[key] = _estimate_density(data, hemi)
    src_den, trg_den = densities['src'], densities['trg']
    _, hemi = zip(*transforms._check_hemi(trg, hemi))

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
    densities = {'src': None, 'trg': None}
    for data, key in zip((src, trg), densities):
        densities[key] = _estimate_density(data, hemi)
    src_den, trg_den = densities['src'], densities['trg']
    _, hemi = zip(*transforms._check_hemi(src, hemi))

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
    densities = {'src': None, 'trg': None}
    for data, key in zip((src, trg), densities):
        densities[key] = _estimate_density(data, hemi)
    src_den, trg_den = densities['src'], densities['trg']

    _, src_hemi = zip(*transforms._check_hemi(src, hemi))
    func = getattr(transforms, f'{src_space.lower()}_to_{alt_space.lower()}')
    src = func(src, src_den, alt_density, hemi=src_hemi, method=method)

    _, trg_hemi = zip(*transforms._check_hemi(trg, hemi))
    func = getattr(transforms, f'{trg_space.lower()}_to_{alt_space.lower()}')
    trg = func(trg, trg_den, alt_density, hemi=trg_hemi, method=method)

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

    return spec


def correlate_images(src, trg, src_space, trg_space, hemi=None,
                     method='linear', resampling='downsample_only',
                     corrtype='pearson', ignore_zero=True,
                     alt_spec=None):
    resamplings = ('downsample_only', 'transform_to_src', 'transform_to_trg',
                   'transform_to_alt')
    if resampling not in resamplings:
        raise ValueError(f'Invalid method: {resampling}')

    opts, err = {}, None
    if resampling == 'transform_to_alt':
        opts['alt_space'], opts['alt_density'] = _check_altspec(alt_spec)
    elif resampling == 'transform_to_src' and src_space.lower() == 'mni152':
        err = ('Specified `src_space` cannot be "MNI152" when `resampling` is '
               '`transform_to_src')
    elif resampling == 'transform_to_trg' and trg_space.lower() == 'mni152':
        err = ('Specified `trg_space` cannot be "MNI152" when `resampling` is '
               '`transform_to_trg`')
    elif resampling == 'transform_to_alt' and opts['alt_space'] == 'mni152':
        err = ('Specified `alt_space` cannot be "MNI152" when `resampling` is '
               '`transform_to_alt`')
    if err is not None:
        raise ValueError(err)

    func = globals()[resampling]

    # resample
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
