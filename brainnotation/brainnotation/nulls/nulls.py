# -*- coding: utf-8 -*-
"""
Contains code for running spatial nulls models
"""

from pathlib import Path

import numpy as np

from brainsmash import mapgen
from brainspace.null_models import moran

from brainnotation.utils import check_fs_subjid
from brainnotation.nulls import burt

SEED = 1234


def load_spins(fn, n_perm=10000):
    """
    Loads spins from `fn`

    Parameters
    ----------
    fn : os.PathLike
        Filepath to file containing spins to load
    n_perm : int, optional
        Number of spins to retain (i.e., subset data)

    Returns
    -------
    spins : (N, P) array_like
        Loaded spins
    """

    npy = Path(fn).with_suffix('.npy')
    if npy.exists():
        spins = np.load(npy, allow_pickle=False, mmap_mode='c')[..., :n_perm]
    else:
        spins = np.loadtxt(fn, delimiter=',', dtype='int32')
        np.save(npy, spins, allow_pickle=False)
        spins = spins[..., :n_perm]

    return spins


def _get_annot(parcellation, scale):
    fetcher = getattr(nndata, f"fetch_{parcellation.replace('atl-', '')}")
    return fetcher('fsaverage5')[scale]


def naive_nonpara(y, n_perm=1000, fn=None):
    y = np.asarray(y)
    rs = np.random.default_rng(SEED)
    if fn is not None:
        spins = load_spins(fn, n_perm=n_perm)
    else:
        spins = np.column_stack([
            rs.permutation(len(y)) for f in range(n_perm)
        ])
    return y[spins]


def vazquez_rodriguez(y, parcellation, scale, n_perm=1000, fn=None):
    y = np.asarray(y)
    if fn is not None:
        spins = load_spins(fn, n_perm=n_perm)
    else:
        if parcellation != 'vertex':
            annot = _get_annot(parcellation, scale)
            coords, hemi = nnsurf.find_parcel_centroids(lhannot=annot.lh,
                                                        rhannot=annot.rh,
                                                        version='fsaverage5',
                                                        surf='sphere',
                                                        method='surface')
        else:
            coords, hemi = nnsurf._get_fsaverage_coords(scale, 'sphere')
        spins = nnstats.gen_spinsamples(coords, hemi, method='original',
                                        n_rotate=n_perm, seed=SEED)
    return y[spins]


def vasa(y, parcellation, scale, n_perm=1000, fn=None):
    y = np.asarray(y)
    if fn is not None:
        spins = load_spins(fn, n_perm=n_perm)
    else:
        annot = _get_annot(parcellation, scale)
        coords, hemi = nnsurf.find_parcel_centroids(lhannot=annot.lh,
                                                    rhannot=annot.rh,
                                                    version='fsaverage5',
                                                    surf='sphere',
                                                    method='surface')
        spins = nnstats.gen_spinsamples(coords, hemi, method='vasa',
                                        n_rotate=n_perm, seed=SEED)
    return y[spins]


def hungarian(y, parcellation, scale, n_perm=1000, fn=None):
    y = np.asarray(y)
    if fn is not None:
        spins = load_spins(fn, n_perm=n_perm)
    else:
        annot = _get_annot(parcellation, scale)
        coords, hemi = nnsurf.find_parcel_centroids(lhannot=annot.lh,
                                                    rhannot=annot.rh,
                                                    version='fsaverage5',
                                                    surf='sphere',
                                                    method='surface')
        spins = nnstats.gen_spinsamples(coords, hemi, method='hungarian',
                                        n_rotate=n_perm, seed=SEED)
    return y[spins]


def baum(y, parcellation, scale, n_perm=1000, fn=None):
    y = np.asarray(y)
    if fn is not None:
        spins = load_spins(fn, n_perm=n_perm)
    else:
        annot = _get_annot(parcellation, scale)
        spins = nnsurf.spin_parcels(lhannot=annot.lh, rhannot=annot.rh,
                                    version='fsaverage5', n_rotate=n_perm,
                                    seed=SEED)
    nulls = y[spins]
    nulls[spins == -1] = np.nan
    return nulls


def cornblath(y, parcellation, scale, n_perm=1000, fn=None):
    y = np.asarray(y)
    annot = _get_annot(parcellation, scale)
    spins = load_spins(fn, n_perm=n_perm) if fn is not None else None
    nulls = nnsurf.spin_data(y, version='fsaverage5', spins=spins,
                             lhannot=annot.lh, rhannot=annot.rh,
                             n_rotate=n_perm, seed=SEED)
    return nulls


def get_distmat(hemi, parcellation, scale):
    if hemi not in ('lh', 'rh'):
        raise ValueError(f'Invalid hemishere designation {hemi}')

    surf = nndata.fetch_fsaverage('fsaverage5')['pial']
    subj, spath = check_fs_subjid('fsaverage5')
    medial = Path(spath) / subj / 'label'
    medial_labels = [
        'unknown', 'corpuscallosum', '???',
        'Background+FreeSurfer_Defined_Medial_Wall'
    ]
    if parcellation == 'vertex':
        medial_path = medial / f'{hemi}.Medial_wall.label'
        dist = surface.get_surface_distance(getattr(surf, hemi),
                                            medial=medial_path,
                                            use_wb=False,
                                            verbose=True)
    else:
        annot = _get_annot(parcellation, scale)
        dist = surface.get_surface_distance(getattr(surf, hemi),
                                            getattr(annot, hemi),
                                            medial_labels=medial_labels,
                                            use_wb=False,
                                            verbose=True)
    return dist


def make_surrogates(data, parcellation, scale, spatnull, n_perm=1000, fn=None):
    if spatnull not in ('burt2018', 'burt2020', 'moran'):
        raise ValueError(f'Cannot make surrogates for null method {spatnull}')

    darr = np.asarray(data)
    dmin = darr[np.logical_not(np.isnan(darr))].min()

    surrogates = np.zeros((len(data), n_perm))
    for n, hemi in enumerate(('lh', 'rh')):
        dist = get_distmat(hemi, parcellation, scale, fn=fn)
        try:
            idx = np.asarray([
                n for n, f in enumerate(data.index)if f.startswith(hemi)
            ])
            hdata = np.squeeze(np.asarray(data.iloc[idx]))
        except AttributeError:
            idx = np.arange(n * (len(data) // 2), (n + 1) * (len(data) // 2))
            hdata = np.squeeze(data[idx])

        # handle NaNs before generating surrogates; should only be relevant
        # when using vertex-level data, but good nonetheless
        mask = np.logical_not(np.isnan(hdata))
        surrogates[idx[np.logical_not(mask)]] = np.nan
        hdata, dist, idx = hdata[mask], dist[np.ix_(mask, mask)], idx[mask]

        if spatnull == 'burt2018':
            # Box-Cox transformation requires positive data
            hdata += np.abs(dmin) + 0.1
            surrogates[idx] = \
                burt.batch_surrogates(dist, hdata, n_surr=n_perm, seed=SEED)
        elif spatnull == 'burt2020':
            if parcellation == 'vertex':
                index = np.argsort(dist, axis=-1)
                dist = np.sort(dist, axis=-1)
                surrogates[idx] = \
                    mapgen.Sampled(hdata, dist, index, seed=SEED)(n_perm).T
            else:
                surrogates[idx] = \
                    mapgen.Base(hdata, dist, seed=SEED)(n_perm, 50).T
        elif spatnull == 'moran':
            dist = dist.astype('float64')  # required for some reason...
            np.fill_diagonal(dist, 1)
            dist **= -1
            mrs = moran.MoranRandomization(joint=True, n_rep=n_perm,
                                           tol=1e-6, random_state=SEED)
            surrogates[idx] = mrs.fit(dist).randomize(hdata).T

    return surrogates
