#!/usr/bin/env python
"""
Script for running MSM-registration on HCP subjects mapping CIVET -> fsLR
"""

from pathlib import Path
import re
import shutil
import time

from joblib import Parallel, delayed
import nibabel as nib
import numpy as np

from neuromaps import images
from neuromaps.utils import run

from neuromaps_dev.civet import register_subject, SPHEREFIX

DATADIR = Path('./data/raw/hcp').resolve()
CIVDIR = Path('./data/raw/civet').resolve()
OUTDIR = Path('./data/derivatives/hcp').resolve()
ATLASDIR = Path('./data/raw/atlases/civet').resolve()
N_PROC = 4


def _regsubj(subdir, affine=None):
    subdir = Path(subdir)
    subid = subdir.name
    subnum = re.search(r'(\d+)', subid).group(1)
    outdir = OUTDIR / subnum / 'surfreg' / 'civet_to_fslr'
    outdir.mkdir(exist_ok=True, parents=True)
    expected = [
        (outdir / f'sub-{subnum}_den-41k_hemi-{hemi}_desc-civettofslr_sphere.'
                  'surf.gii')
        for hemi in ('L', 'R')
    ]

    if any(not fn.exists() for fn in expected):
        print(f'{time.ctime()}: Registering {subid} to fsLR')
        out = register_subject(subdir, affine=affine, only_gen_affine=False)
        for (fn, mv) in zip(out, expected):
            shutil.move(fn, mv)

    return expected


def main():
    OUTDIR.mkdir(exist_ok=True, parents=True)
    subjects = np.loadtxt(DATADIR / 'subjects.txt', dtype=str)
    missing = np.loadtxt(DATADIR / 'missing.txt', dtype=str)
    subjects = np.core.defchararray.add(
        'HCP_', np.setdiff1d(subjects, missing)
    )

    # generate rotational affines first
    affines = Parallel(n_jobs=N_PROC)(
        delayed(register_subject)(CIVDIR / sub, affine=None,
                                  only_gen_affine=True)
        for sub in subjects
    )

    # take average of affines and use to seed final alignment
    final = (CIVDIR / 'left_affine.txt', CIVDIR / 'right_affine.txt')
    for n, affs in enumerate(zip(*affines)):
        rot = np.mean([np.loadtxt(aff) for aff in affs], axis=0)
        np.savetxt(final[n], rot, fmt='%.10f')

    # now generate the final aligned spherical surfaces
    spheres = Parallel(n_jobs=N_PROC)(
        delayed(_regsubj)(CIVDIR / sub, affine=final) for sub in subjects
    )

    # create the average spheres (across subjects)
    for surfs, hemi in zip(zip(*spheres), ('L', 'R')):
        sphere = images.average_surfaces(*surfs)
        fn = f'tpl-civet_space-fsLR_den-41k_hemi-{hemi}_sphere.surf.gii'
        nib.save(sphere, ATLASDIR / fn)
        run(SPHEREFIX.format(sphererot=ATLASDIR / fn), quiet=True)

    # also calculate midthickness average vertex areas (for use w/resampling)
    for hemi in ('left', 'right'):
        vaavg = np.mean([
            images.vertex_areas(
                CIVDIR / sub / f'{sub}_mid_surface_rsl_{hemi}_81920.surf.gii'
            ) for sub in subjects
        ], axis=0)
        fn = (f'tpl-civet_space-fsLR_den-41k_hemi-{hemi}_'
              'desc-vaavg_midthickness.surf.gii')
        nib.save(images.construct_shape_gii(vaavg), ATLASDIR / fn)


if __name__ == "__main__":
    main()
