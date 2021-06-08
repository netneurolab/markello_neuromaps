# -*- coding: utf-8 -*-
"""
Functions for comparing data
"""

from brainnotation import transforms
from brainnotation.datasets import ALIAS, DENSITIES
from brainnotation.images import load_gifti


_resampling_docs = dict(
    resample_in="""\
{src, trg} : str or os.PathLike or niimg_like or nib.GiftiImage or tuple
    Input data to be resampled to each other
{src,trg}_space : str
    Template space of {`src`, `trg`} data\
method : {'nearest', 'linear'}, optional
    Method for resampling. Specify 'nearest' if `data` are label images.
    Default: 'linear'\
""",
    hemi="""\
hemi : {'L', 'R'}, optional
    If `src` and `trg` are not tuples this specifies the hemisphere the data
    represent. Default: None\
""",
    resample_out="""\
src, trg : tuple-of-nib.GiftiImage
    Resampled images\
"""
)


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


def downsample_only(src, trg, src_space, trg_space, method='linear',
                    hemi=None):
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
{hemi}

Returns
-------
{resample_out}
""".format(**_resampling_docs)


def transform_to_src(src, trg, src_space, trg_space, method='linear',
                     hemi=None):
    src_den, trg_den = _estimate_density((src, trg), hemi)

    func = getattr(transforms, f'{trg_space.lower()}_to_{src_space.lower()}')
    trg = func(trg, trg_den, src_den, hemi=hemi, method=method)

    return src, trg


transform_to_src.__doc__ = """\
Resamples `trg` to match space and density of `src`

Parameters
----------
{resample_in}
{hemi}

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


def transform_to_alt(src, trg, src_space, trg_space, method='linear',
                     hemi=None, alt_space='fsaverage', alt_density='41k'):
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
{hemi}
alt_space : {{'fsaverage', 'fsLR', 'civet'}}, optional
    Alternative space to which `src` and `trg` should be transformed. Default:
    'fsaverage'
alt_density : str, optional
    Resolution to which `src` and `trg` should be resampled. Must be valid
    with `alt_space`. Default: '41k'

Returns
-------
{resample_out}
""".format(**_resampling_docs)


def mni_transformation(src, trg, src_space, trg_space, method='linear'):
    if src_space != 'MNI152':
        raise ValueError('Cannot perform MNI transformation when src_space is '
                         f'not "MNI152." Received: {src_space}.')
    trg_den = trg
    if trg_space != 'MNI152':
        trg_den, = _estimate_density((trg_den,), None)
    func = getattr(transforms, f'mni152_to_{trg_space.lower()}')
    src = func(src, trg_den, method=method)
    return src, trg


mni_transformation.__doc__ = """\
Resamples `src` in MNI152 to `trg` space

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

    return (ALIAS.get(spec[0], spec[0]), spec[1])


def resample_images(src, trg, src_space, trg_space, method='linear',
                    hemi=None, resampling='downsample_only', alt_spec=None):

    resamplings = ('downsample_only', 'transform_to_src', 'transform_to_trg',
                   'transform_to_alt')
    if resampling not in resamplings:
        raise ValueError(f'Invalid method: {resampling}')

    src_space = ALIAS.get(src_space, src_space)
    trg_space = ALIAS.get(trg_space, trg_space)

    # all this input handling just to deal with volumetric images :face_palm:
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
        src, trg = mni_transformation(src, trg, src_space, trg_space, method)
    elif trg_space == 'MNI152':
        trg, src = mni_transformation(trg, src, trg_space, src_space, method)
    else:
        func = globals()[resampling]
        src, trg = func(src, trg, src_space, trg_space, hemi=hemi,
                        method=method, **opts)

    return tuple(load_gifti(s) for s in src), tuple(load_gifti(t) for t in trg)


resample_images.__doc__ = """\
Correlates images `src` and `trg`, resampling as needed

Parameters
----------
{resample_in}
{hemi}
resampling : str, optional
    Name of resampling function to resample `src` and `trg`. Must be one of:
    {{'downsample_only', 'transform_to_src', 'transform_to_trg',
    'transform_to_alt'}}. See Notes for more info. Default: 'downsample_only'
corrtype : {{'pearson', 'spearman'}}, optional
    Type of correlation to perform. Default: 'pearson'
ignore_zero : bool, optional
    Whether to perform correlations ignoring all zero values in `src` and
    `trg` data. Default: True
alt_spec : (2,) tuple-of-str
    Where entries are (space, density) of desired target space. Only used if
    `resampling='transform_to_alt'. Default: None

Returns
-------
{resample_out}

Notes
-----
The four available `resampling` strategies will control how `src` and/or `trg`
are resampled prior to correlation. Options include:

    1. `resampling='downsample_only'`

    Data from `src` and `trg` are resampled to the lower resolution of the two
    input datasets

    2. `resampling='transform_to_src'`

    Data from `trg` are always resampled to match `src` space and resolution

    3. `resampling='transform_to_trg'`

    Data from `src` are always resampled to match `trg` space and resolution

    4. `resampling='transform_to_alt'`

    Data from `trg` and `src` are resampled to the space and resolution
    specified by `alt_spec` (space, density)
""".format(**_resampling_docs)