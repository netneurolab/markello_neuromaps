#!/usr/bin/env python
"""
Script for generating 4k + 8k average vertex areas for fsLR mesh
"""

from pathlib import Path
import time

from joblib import Parallel, delayed
import nibabel as nib
import numpy as np

from neuromaps.images import construct_shape_gii, vertex_areas
from neuromaps.utils import tmpname, run

ATLASDIR = Path('./data/raw/atlases/fsLR').resolve()
DATADIR = Path('./data/raw/hcp').resolve()
SURFRESAMPLE = 'wb_command -surface-resample {surface} {src} {trg} ' \
               'BARYCENTRIC {out}'
N_PROC = 1


def resample(subdir, res):
    if res not in ('4k', '8k'):
        raise ValueError(f'Invalid res: {res}')

    subdir = Path(subdir) / 'MNINonLinear'
    subid = subdir.parent.name
    print(f'{time.ctime()}: Generating {res} midthick for {subid}')

    areas = ()
    for hemi in 'LR':
        src = ATLASDIR / f'tpl-fsLR_den-164k_hemi-{hemi}_sphere.surf.gii'
        trg = ATLASDIR / f'tpl-fsLR_den-{res}_hemi-{hemi}_sphere.surf.gii'
        midthick = subdir / f'{subid}.{hemi}.sphere.164k_fs_LR.surf.gii'
        out = tmpname('.surf.gii')
        run(SURFRESAMPLE.format(surface=midthick, src=src, trg=trg, out=out))
        areas += (vertex_areas(out), )
        out.unlink()

    return areas


def main():
    subjects = np.loadtxt(DATADIR / 'subjects.txt', dtype=str)
    missing = np.loadtxt(DATADIR / 'missing.txt', dtype=str)
    subjects = np.setdiff1d(subjects, missing)

    for res in ('4k', '8k'):
        areas = Parallel(n_jobs=N_PROC)(
            delayed(resample)(DATADIR / sub, res) for sub in subjects
        )
        for data, hemi in zip(zip(*areas), ('L', 'R')):
            fn = f'tpl-fsLR_den-{res}_hemi-{hemi}_desc-vaavg_midthickness.' \
                 'shape.gii'
            nib.save(construct_shape_gii(np.mean(data, axis=0)), ATLASDIR / fn)


if __name__ == "__main__":
    main()
