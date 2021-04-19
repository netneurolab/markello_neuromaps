#!/usr/bin/env python
"""
Script for running registration-fusion on HCP subjects
"""

from pathlib import Path
import shutil
import time

from joblib import Parallel, delayed
import numpy as np

from brainnotation.regfusion import fs_regfusion, hcp_regfusion

DATADIR = Path('./data/raw/hcp').resolve()
OUTDIR = Path('./data/derivatives/hcp').resolve()
REGFUNCS = dict(
    freesurfer=fs_regfusion,
    hcp=hcp_regfusion
)


def _regfusion(subdir, method):
    if method not in REGFUNCS:
        raise ValueError(f'Invalid regfusion method: {method}')

    subdir = Path(subdir)
    outdir = OUTDIR / subdir.name / 'regfusion' / method
    outdir.mkdir(exist_ok=True, parents=True)
    expected = [
        outdir / f'sub-{subdir.name}_hemi-{hemi}_desc-{key}_index.func.gii'
        for hemi in ('lh', 'rh') for key in ('x', 'y', 'z')
    ]

    # run reg-fusion only if it hasn't been run before
    if any(not fn.exists() for fn in expected):
        print(f'{time.ctime()}: Running {method} reg-fusion {subdir.name}')
        out = REGFUNCS[method](subdir, verbose=False)
        for key, imgs in out.items():
            for hemi, img in zip(('lh', 'rh'), imgs):
                fn = f'sub-{subdir.name}_hemi-{hemi}_desc-{key}_index.func.gii'
                shutil.move(img, outdir / fn)


def main():
    OUTDIR.mkdir(exist_ok=True, parents=True)
    subjects = np.loadtxt(DATADIR / 'subjects.txt', dtype=str)
    missing = np.loadtxt(DATADIR / 'missing.txt', dtype=str)
    subjects = np.setdiff1d(subjects, missing)

    pool, fusion = Parallel(n_jobs=6), delayed(_regfusion)
    for method in REGFUNCS:
        pool(fusion(DATADIR / sub, method=method) for sub in subjects)


if __name__ == "__main__":
    main()
