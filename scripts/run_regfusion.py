#!/usr/bin/env python
"""
Script for running registration-fusion on HCP subjects
"""

from pathlib import Path
import shutil
import time

from joblib import Parallel, delayed
import numpy as np

from brainnotation_dev.regfusion import (fs_regfusion, hcp_regfusion,
                                         civet_regfusion, get_files)

DATADIR = Path('./data/raw/hcp').resolve()
CIVDIR = Path('./data/raw/civet').resolve()
OUTDIR = Path('./data/derivatives/hcp').resolve()
N_PROC = 4
REGFUNCS = dict(
    freesurfer=fs_regfusion,
    hcp=hcp_regfusion,
    civet=civet_regfusion,
)
RESOLUTIONS = dict(
    freesurfer=[f'fsaverage{n}' for n in ('', '3', '4', '5', '6')],
    hcp=['32k', '164k'],
    civet=['41k']
)
MAPPING = dict(
    fsaverage3='1k',
    fsaverage4='3k',
    fsaverage5='10k',
    fsaverage6='41k',
    fsaverage='164k',
)


def _regfusion(subdir, method, resolution):
    if method not in REGFUNCS:
        raise ValueError(f'Invalid regfusion method: {method}')

    subdir = Path(subdir)
    subid = subdir.name
    outdir = OUTDIR / subid / 'regfusion' / method
    outdir.mkdir(exist_ok=True, parents=True)
    res = MAPPING.get(resolution, resolution)
    expected = [
        (outdir / f'sub-{subid}_den-{res}_hemi-{hemi}_desc-{key}_index.'
                  'func.gii')
        for hemi in ('L', 'R') for key in ('x', 'y', 'z')
    ]

    if method == 'civet':
        subdir = CIVDIR / ('HCP_' + subid)

    # run reg-fusion only if it hasn't been run before
    if any(not fn.exists() for fn in expected):
        print(f'{time.ctime()}: Running {method} reg-fusion {subid} '
              f'for {resolution} resolution')
        out = REGFUNCS[method](subdir, res=resolution, verbose=False)
        for key, imgs in out.items():
            for hemi, img in zip(('L', 'R'), imgs):
                fn = f'sub-{subid}_den-{res}_hemi-{hemi}_desc-{key}_' \
                     f'index.func.gii'
                shutil.move(img, outdir / fn)


def main():
    OUTDIR.mkdir(exist_ok=True, parents=True)
    subjects = np.loadtxt(DATADIR / 'subjects.txt', dtype=str)
    missing = np.loadtxt(DATADIR / 'missing.txt', dtype=str)
    subjects = np.setdiff1d(subjects, missing)

    pool, fusion = Parallel(n_jobs=N_PROC), delayed(_regfusion)
    # get files first
    pool(delayed(get_files)(sub, OUTDIR, verbose=False) for sub in subjects)
    # now run reg-fusion for all the different formats
    for method in REGFUNCS:
        for resolution in RESOLUTIONS[method]:
            pool(fusion(DATADIR / sub, method, resolution) for sub in subjects)


if __name__ == "__main__":
    main()
