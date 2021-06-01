# -*- coding: utf-8 -*-
"""
Functions for fetching datasets (from the internet, if necessary)
"""

from collections import namedtuple

from nilearn.datasets.utils import _fetch_files
from sklearn.utils import Bunch

from brainnotation.datasets.utils import _get_data_dir, _get_dataset_info

SURFACE = namedtuple('Surface', ('lh', 'rh'))
DENSITIES = dict(
    civet=['41k', '164k'],
    fsaverage=['3k', '10k', '41k', '164k'],
    fsLR=['4k', '8k', '32k', '164k'],
    MNI152=['1mm', '2mm', '3mm']
)


def _fetch_atlas(atlas, density, keys, url=None, data_dir=None, verbose=1):
    """ Helper function to get requested `atlas`
    """

    densities = DENSITIES[atlas]
    if density not in densities:
        raise ValueError(f'Invalid density: {density}. Must be one of '
                         f'{densities}')

    data_dir = _get_data_dir(data_dir=data_dir)
    info = _get_dataset_info(atlas)[density]
    if url is None:
        url = info['url']
    opts = {
        'uncompress': True,
        'md5sum': info['md5'],
        'move': f'{atlas}{density}.tar.gz'
    }

    if atlas == 'MNI152':
        filenames = [
            f'atlases/{atlas}/tpl-MNI152NLin2009cAsym_res-{density}_T1w.nii.gz'
        ]
    else:
        filenames = [
            f'atlases/{atlas}/tpl-{atlas}_den-{{}}_hemi-{{}}_{{}}.surf.gii'
            .format(density, hemi, res) for res in keys for hemi in ['L', 'R']
        ]

    data = _fetch_files(data_dir, files=[(f, url, opts) for f in filenames],
                        verbose=verbose)
    if atlas != 'MNI152':
        data = [SURFACE(*data[i:i + 2]) for i in range(0, len(data), 2)]

    return Bunch(**dict(zip(keys, data)))


def fetch_civet(density='41k', url=None, data_dir=None, verbose=1):
    keys = ['white', 'midthickness', 'inflated', 'veryinflated', 'sphere']
    return _fetch_atlas('civet', density, keys,
                        url=url, data_dir=data_dir, verbose=verbose)


def fetch_fsaverage(density='41k', url=None, data_dir=None, verbose=1):
    keys = ['white', 'pial', 'inflated', 'sphere']
    return _fetch_atlas('fsaverage', density, keys,
                        url=url, data_dir=data_dir, verbose=verbose)


def fetch_fslr(density='32k', url=None, data_dir=None, verbose=1):
    keys = ['midthickness', 'inflated', 'veryinflated', 'sphere']
    if density in ('4k', '8k'):
        keys.remove('veryinflated')
    return _fetch_atlas('fsLR', density, keys,
                        url=url, data_dir=data_dir, verbose=verbose)


def fetch_mni152(density='1mm', url=None, data_dir=None, verbose=1):
    keys = ['T1w']
    return _fetch_atlas('MNI152', density, keys,
                        url=url, data_dir=data_dir, verbose=verbose)


def fetch_regfusion(space, url=None, data_dir=None, verbose=1):
    if space not in DENSITIES:
        raise ValueError(f'Invalid space: {space}')
    densities = DENSITIES[space]

    data_dir = _get_data_dir(data_dir=data_dir)
    info = _get_dataset_info('regfusion')
    if url is None:
        url = info['url']
    opts = {
        'uncompress': True,
        'md5sum': info['md5'],
        'move': 'regfusion.tar.gz'
    }

    filenames = [
        'atlases/regfusion/'
        f'tpl-MNI152_space-{space}_den-{{}}_hemi-{{}}_regfusion.txt'
        .format(density, hemi)
        for density in densities for hemi in ['L', 'R']
    ]

    data = _fetch_files(data_dir, files=[(f, url, opts) for f in filenames],
                        verbose=verbose)
    data = [SURFACE(*data[i:i + 2]) for i in range(0, len(data), 2)]

    return Bunch(**dict(zip(densities, data)))


def fetch_all_atlases(data_dir=None, verbose=1):
    for key, resolutions in DENSITIES.items():
        for res in resolutions:
            fetcher = getattr(globals(), f'fetch_{key}')
            fetcher(res, data_dir=data_dir, verbose=verbose)
        fetch_regfusion(key, data_dir=data_dir, verbose=verbose)
