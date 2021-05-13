# -*- coding: utf-8 -*-
"""
Contains code for running spatial nulls models
"""

from pathlib import Path

import numpy as np
import pandas as pd

from brainsmash import mapgen
from brainspace.null_models import moran
from netneurotools import (datasets as nndata,
                           freesurfer as nnsurf,
                           stats as nnstats)
from parspin import burt, simnulls, surface

FIGDIR = Path('./figures/supplementary/comp_time').resolve()
ROIDIR = Path('./data/raw/rois').resolve()
SPDIR = Path('./data/derivatives/spins').resolve()
SIMDIR = Path('./data/derivatives/simulated').resolve()
DISTDIR = Path('./data/derivatives/geodesic').resolve()
OUTDIR = Path('./data/derivatives/supplementary/comp_time').resolve()

N_PERM = 1000
SEED = 1234
USE_CACHED = True


def _get_annot(parcellation, scale):
    fetcher = getattr(nndata, f"fetch_{parcellation.replace('atl-', '')}")
    return fetcher('fsaverage5', data_dir=ROIDIR)[scale]


def naive_nonpara(y, fn=None):
    y = np.asarray(y)
    rs = np.random.default_rng(SEED)
    if USE_CACHED and fn is not None:
        spins = simnulls.load_spins(fn, n_perm=N_PERM)
    else:
        spins = np.column_stack([
            rs.permutation(len(y)) for f in range(N_PERM)
        ])
    return y[spins]


def vazquez_rodriguez(y, parcellation, scale, fn=None):
    y = np.asarray(y)
    if USE_CACHED and fn is not None:
        spins = simnulls.load_spins(fn, n_perm=N_PERM)
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
                                        n_rotate=N_PERM, seed=SEED)
    return y[spins]


def vasa(y, parcellation, scale, fn=None):
    y = np.asarray(y)
    if USE_CACHED and fn is not None:
        spins = simnulls.load_spins(fn, n_perm=N_PERM)
    else:
        annot = _get_annot(parcellation, scale)
        coords, hemi = nnsurf.find_parcel_centroids(lhannot=annot.lh,
                                                    rhannot=annot.rh,
                                                    version='fsaverage5',
                                                    surf='sphere',
                                                    method='surface')
        spins = nnstats.gen_spinsamples(coords, hemi, method='vasa',
                                        n_rotate=N_PERM, seed=SEED)
    return y[spins]


def hungarian(y, parcellation, scale, fn=None):
    y = np.asarray(y)
    if USE_CACHED and fn is not None:
        spins = simnulls.load_spins(fn, n_perm=N_PERM)
    else:
        annot = _get_annot(parcellation, scale)
        coords, hemi = nnsurf.find_parcel_centroids(lhannot=annot.lh,
                                                    rhannot=annot.rh,
                                                    version='fsaverage5',
                                                    surf='sphere',
                                                    method='surface')
        spins = nnstats.gen_spinsamples(coords, hemi, method='hungarian',
                                        n_rotate=N_PERM, seed=SEED)
    return y[spins]


def baum(y, parcellation, scale, fn=None):
    y = np.asarray(y)
    if USE_CACHED and fn is not None:
        spins = simnulls.load_spins(fn, n_perm=N_PERM)
    else:
        annot = _get_annot(parcellation, scale)
        spins = nnsurf.spin_parcels(lhannot=annot.lh, rhannot=annot.rh,
                                    version='fsaverage5', n_rotate=N_PERM,
                                    seed=SEED)
    nulls = y[spins]
    nulls[spins == -1] = np.nan
    return nulls


def cornblath(y, parcellation, scale, fn=None):
    y = np.asarray(y)
    annot = _get_annot(parcellation, scale)
    spins = simnulls.load_spins(fn, n_perm=N_PERM) if USE_CACHED else None
    nulls = nnsurf.spin_data(y, version='fsaverage5', spins=spins,
                             lhannot=annot.lh, rhannot=annot.rh,
                             n_rotate=N_PERM, seed=SEED)
    return nulls


def get_distmat(hemi, parcellation, scale, fn=None):
    if hemi not in ('lh', 'rh'):
        raise ValueError(f'Invalid hemishere designation {hemi}')

    if USE_CACHED and fn is not None:
        fn = DISTDIR / parcellation / 'nomedial' / f'{scale}_{hemi}_dist.npy'
        dist = np.load(fn, allow_pickle=False, mmap_mode='c').astype('float32')
    else:
        surf = nndata.fetch_fsaverage('fsaverage5', data_dir=ROIDIR)['pial']
        subj, spath = nnsurf.check_fs_subjid('fsaverage5')
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


def make_surrogates(data, parcellation, scale, spatnull, fn=None):
    if spatnull not in ('burt2018', 'burt2020', 'moran'):
        raise ValueError(f'Cannot make surrogates for null method {spatnull}')

    darr = np.asarray(data)
    dmin = darr[np.logical_not(np.isnan(darr))].min()

    surrogates = np.zeros((len(data), N_PERM))
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
                burt.batch_surrogates(dist, hdata, n_surr=N_PERM, seed=SEED)
        elif spatnull == 'burt2020':
            if parcellation == 'vertex':
                index = np.argsort(dist, axis=-1)
                dist = np.sort(dist, axis=-1)
                surrogates[idx] = \
                    mapgen.Sampled(hdata, dist, index, seed=SEED)(N_PERM).T
            else:
                surrogates[idx] = \
                    mapgen.Base(hdata, dist, seed=SEED)(N_PERM, 50).T
        elif spatnull == 'moran':
            dist = dist.astype('float64')  # required for some reason...
            np.fill_diagonal(dist, 1)
            dist **= -1
            mrs = moran.MoranRandomization(joint=True, n_rep=N_PERM,
                                           tol=1e-6, random_state=SEED)
            surrogates[idx] = mrs.fit(dist).randomize(hdata).T

    return surrogates
