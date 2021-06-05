# -*- coding: utf-8 -*-
"""
Contains code for running spatial nulls models
"""

import numpy as np

from brainsmash import mapgen
from brainspace.null_models.moran import MoranRandomization

from brainnotation.datasets import fetch_atlas
from brainnotation.images import load_gifti
from brainnotation.points import get_surface_distance
from brainnotation.nulls import burt
from brainnotation.nulls.utils import (gen_spinsamples, get_parcel_centroids,
                                       relabel_gifti, spin_data, spin_parcels,
                                       PARCIGNORE)


def naive_nonpara(y, n_perm=1000, seed=None):
    y = np.asarray(y)
    rs = np.random.default_rng(seed)
    spins = np.column_stack([rs.permutation(len(y)) for f in range(n_perm)])
    return y[spins]


def alexander_bloch(y, atlas='fsaverage', density='10k', parcellation=None,
                    n_perm=1000, seed=None):
    y = np.asarray(y)
    surfaces = fetch_atlas(atlas, density)['sphere']
    coords, hemi = get_parcel_centroids(surfaces,
                                        parcellation=parcellation,
                                        method='surface')
    spins = gen_spinsamples(coords, hemi, method='original',
                            n_rotate=n_perm, seed=seed)
    return y[spins]


vazquez_rodriguez = alexander_bloch


def vasa(y, atlas='fsaverage', density='10k', parcellation=None, n_perm=1000,
         seed=None):
    y = np.asarray(y)
    surfaces = fetch_atlas(atlas, density)['sphere']
    coords, hemi = get_parcel_centroids(surfaces,
                                        parcellation=parcellation,
                                        method='surface')
    spins = gen_spinsamples(coords, hemi, method='vasa',
                            n_rotate=n_perm, seed=seed)
    return y[spins]


def hungarian(y, atlas='fsaverage', density='10k', parcellation=None,
              n_perm=1000, seed=None):
    y = np.asarray(y)
    surfaces = fetch_atlas(atlas, density)['sphere']
    coords, hemi = get_parcel_centroids(surfaces,
                                        parcellation=parcellation,
                                        method='surface')
    spins = gen_spinsamples(coords, hemi, method='hungarian',
                            n_rotate=n_perm, seed=seed)
    return y[spins]


def baum(y, atlas='fsaverage', density='10k', parcellation=None, n_perm=1000,
         seed=None):
    y = np.asarray(y)
    surfaces = fetch_atlas(atlas, density)['sphere']
    spins = spin_parcels(surfaces, parcellation=parcellation,
                         n_rotate=n_perm, seed=seed)
    nulls = y[spins]
    nulls[spins == -1] = np.nan
    return nulls


def cornblath(y, atlas='fsaverage', density='10k', parcellation=None,
              n_perm=1000, seed=None):
    y = np.asarray(y)
    surfaces = fetch_atlas(atlas, density)['sphere']
    nulls = spin_data(y, surfaces, parcellation,
                      n_rotate=n_perm, seed=seed)
    return nulls


def get_distmat(hemi, atlas='fsaverage', density='10k', parcellation=None,
                drop=None):
    if hemi not in ('L', 'R'):
        raise ValueError(f'Invalid hemishere designation {hemi}')

    if drop is None:
        drop = PARCIGNORE

    atlas = fetch_atlas(atlas, density)
    surf, medial = getattr(atlas['pial'], hemi), getattr(atlas['medial'], hemi)
    if parcellation is None:
        dist = get_surface_distance(surf, medial=medial)
    else:
        dist = get_surface_distance(surf, parcellation=parcellation,
                                    medial_labels=drop, drop=drop)
    return dist


def _make_surrogates(data, method, atlas='fsaverage', density='10k',
                     parcellation=None, n_perm=1000, seed=None):
    if method not in ('burt2018', 'burt2020', 'moran'):
        raise ValueError(f'Invalid null method: {method}')

    darr = np.asarray(data)
    dmin = darr[np.logical_not(np.isnan(darr))].min()
    parcellation = relabel_gifti(parcellation)

    surrogates = np.zeros((len(data), n_perm))
    for n, hemi in enumerate(('L', 'R')):
        dist = get_distmat(hemi, atlas=atlas, density=density,
                           parcellation=parcellation[n])

        if parcellation is None:
            idx = np.arange(n * (len(data) // 2), (n + 1) * (len(data) // 2))
        else:
            idx = np.unique(load_gifti(parcellation[n]).agg_data())[1:]

        hdata = np.squeeze(data[idx])
        mask = np.logical_not(np.isnan(hdata))
        surrogates[idx[np.logical_not(mask)]] = np.nan
        hdata, dist, idx = hdata[mask], dist[np.ix_(mask, mask)], idx[mask]

        if method == 'burt2018':
            hdata += np.abs(dmin) + 0.1
            surrogates[idx] = \
                burt.batch_surrogates(dist, hdata, n_surr=n_perm, seed=seed)
        elif method == 'burt2020':
            if parcellation is None:
                index = np.argsort(dist, axis=-1)
                dist = np.sort(dist, axis=-1)
                surrogates[idx] = \
                    mapgen.Sampled(hdata, dist, index, seed=seed)(n_perm).T
            else:
                surrogates[idx] = \
                    mapgen.Base(hdata, dist, seed=seed)(n_perm, 50).T
        elif method == 'moran':
            dist = dist.astype('float64')
            np.fill_diagonal(dist, 1)
            dist **= -1
            mrs = MoranRandomization(joint=True, n_rep=n_perm, tol=1e-6,
                                     random_state=seed)
            surrogates[idx] = mrs.fit(dist).randomize(hdata).T

    return surrogates


def burt2018(y, atlas='fsaverage', density='10k', parcellation=None,
             n_perm=1000, seed=None):
    return _make_surrogates(y, 'burt2018', atlas=atlas, density=density,
                            parcellation=parcellation, n_perm=n_perm,
                            seed=seed)


def burt2020(y, atlas='fsaverage', density='10k', parcellation=None,
             n_perm=1000, seed=None):
    return _make_surrogates(y, 'burt2020', atlas=atlas, density=density,
                            parcellation=parcellation, n_perm=n_perm,
                            seed=seed)


def moran(y, atlas='fsaverage', density='10k', parcellation=None, n_perm=1000,
          seed=None):
    return _make_surrogates(y, 'moran', atlas=atlas, density=density,
                            parcellation=parcellation, n_perm=n_perm,
                            seed=seed)
