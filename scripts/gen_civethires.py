#!/usr/bin/env python
"""
Script for generating hi-res meshes from CIVET outputs
"""

from pathlib import Path
import re
import time

from joblib import Parallel, delayed
import numpy as np

from brainnotation.images import obj_to_gifti
from netneurotools.utils import run

DATADIR = Path('./data/raw/hcp').resolve()
CIVDIR = Path('./data/raw/civet').resolve()
OUTDIR = Path('./data/derivatives/civet').resolve()
ATLASDIR = Path('./data/raw/atlases/civet').resolve()
MSMSULC = 'sub-{subnum}_den-{den}_hemi-{hemi}_desc-civettofslr_sphere.surf.gii'
SPHEREPROJECT = 'wb_command -surface-sphere-project-unproject {spherein} ' \
                '{project} {unproject} {sphereout}'
SURFRESAMPLE = 'wb_command -surface-resample {surface} {src} {trg} ' \
               'BARYCENTRIC {out}'
HEMI = dict(L='left', R='right')
N_PROC = 4


def make_hires_subject(subdir, verbose=True):
    subdir = Path(subdir)
    subid = subdir.name
    subnum = re.search(r'(\d+)', subid).group(1)
    subdir = subdir / 'gifti'
    print(f'{time.ctime()}: Generating hi-res CIVET mesh for {subnum}')

    for hemi in ('L', 'R'):
        fmt = f'{subid}_{{surf}}_surface_rsl_{HEMI[hemi]}_{{res}}.obj'

        # hi-res template sphere
        tpl = ATLASDIR / f'tpl-civet_den-164k_hemi-{hemi}_sphere.surf.gii'

        # lo-res native sphere
        proj = subdir / fmt.format(surf='sphere', res='81920')
        # low-res native midthickness
        fn = subdir.parent / 'surfaces' / fmt.format(sphere='mid', res='81920')
        midthick = subdir / fn.name.replace('.obj', '.surf.gii')
        if not midthick.exists():
            obj_to_gifti(fn, midthick)
        # hi-res native midthickness (to-be-generated)
        midout = subdir / midthick.name.replace('81920', '327684')

        hemi = HEMI.get(hemi)
        # lo-res MSM sphere
        msm = (DATADIR / subnum / 'surfreg' / 'civet_to_fslr'
               / MSMSULC.format(subnum=subnum, den='41k', hemi=hemi))
        # hi-res MSM sphere (to-be-generated)
        msmout = (DATADIR / subnum / 'surfreg' / 'civet_to_fslr'
                  / MSMSULC.format(subnum=subnum, den='164k', hemi=hemi))

        if not msm.exists():
            print(f'WARNING: {subnum} missing final MSM output; skipping')
            return

        # gen hi-res MSM rotated sphere (i.e., CIVET hi-res align with fsLR)
        if not msmout.exists():
            run(SPHEREPROJECT.format(spherein=tpl, project=proj,
                                     unproject=msm, sphereout=msmout),
                quiet=not verbose)

        # then resample the midthickness to hi-res surface
        if not midout.exists():
            run(SURFRESAMPLE.format(surface=midthick, src=msm,
                                    trg=msmout, out=midout),
                quiet=not verbose)


def main():
    OUTDIR.mkdir(exist_ok=True, parents=True)
    subjects = np.loadtxt(DATADIR / 'subjects.txt', dtype=str)
    missing = np.loadtxt(DATADIR / 'missing.txt', dtype=str)
    subjects = np.core.defchararray.add(
        'HCP_', np.setdiff1d(subjects, missing)
    )

    Parallel(n_jobs=N_PROC)(
        delayed(make_hires_subject)(CIVDIR / sub) for sub in subjects
    )


if __name__ == "__main__":
    main()
