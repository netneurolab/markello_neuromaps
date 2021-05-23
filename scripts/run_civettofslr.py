#!/usr/bin/env python
"""
Script for running MSM-registration on HCP subjects mapping CIVET -> fsLR
"""

from pathlib import Path
import re
import shutil
import time

from joblib import Parallel, delayed
import numpy as np

from brainnotation.civet import register_subject

DATADIR = Path('./data/raw/hcp').resolve()
CIVDIR = Path('./data/raw/civet').resolve()
OUTDIR = Path('./data/derivatives/hcp').resolve()


def _regsubj(subdir, affine=None):
    subdir = Path(subdir)
    subid = subdir.name
    subnum = re.search(r'(\d+)', subid).group(1)
    outdir = OUTDIR / subnum / 'surfreg' / 'civet_to_fslr'
    outdir.mkdir(exist_ok=True, parents=True)
    expected = [
        (outdir / f'sub-{subnum}_den-41k_hemi-{hemi}_desc-civettofslr_sphere.'
                  'surf.gii')
        for hemi in ('lh', 'rh')
    ]

    if any(not fn.exists() for fn in expected):
        print(f'{time.ctime()}: Registering {subid} to fsLR')
        out = register_subject(subdir, affine=affine, only_gen_affine=False)
        for (fn, mv) in zip(out, expected):
            shutil.move(fn, mv)


def main():
    OUTDIR.mkdir(exist_ok=True, parents=True)
    subjects = np.loadtxt(DATADIR / 'subjects.txt', dtype=str)
    missing = np.loadtxt(DATADIR / 'missing.txt', dtype=str)
    subjects = np.core.defchararray.add(
        'HCP_', np.setdiff1d(subjects, missing)
    )

    # generate rotational affines first
    affines = Parallel(n_jobs=4)(
        delayed(register_subject)(CIVDIR / sub, affine=None,
                                  only_gen_affine=True)
        for sub in subjects
    )

    # take average of affines and use to seed final alignment
    final = ()
    for affs, hemi in zip(zip(*affines), ('left', 'right')):
        rot = np.mean([np.loadtxt(aff) for aff in affs], axis=0)
        final += (CIVDIR / f'{hemi}_affine.txt',)
        np.savetxt(final[-1], rot, fmt='%.10f')

    # now generate the final aligned spherical surfaces
    Parallel(n_jobs=4)(
        delayed(_regsubj)(CIVDIR / sub, affine=final) for sub in subjects
    )


if __name__ == "__main__":
    main()
