#!/usr/bin/env python
"""
Script for generating hi-res meshes from CIVET outputs
"""

from pathlib import Path
import re
import time

from joblib import Parallel, delayed
import nibabel as nib
import numpy as np

from brainnotation import images
from brainnotation.utils import run

from brainnotation_dev.civet import SPHEREFIX

DATADIR = Path('./data/raw/hcp').resolve()
CIVDIR = Path('./data/raw/civet').resolve()
ATLASDIR = Path('./data/raw/atlases/civet').resolve()
MSMSULC = 'sub-{subnum}_den-{den}_hemi-{hemi}_desc-civettofslr_sphere.surf.gii'
SPHEREPROJECT = 'wb_command -surface-sphere-project-unproject {spherein} ' \
                '{project} {unproject} {sphereout}'
SURFRESAMPLE = 'wb_command -surface-resample {surface} {src} {trg} ' \
               'BARYCENTRIC {out}'
HEMI = dict(L='left', R='right')
N_PROC = 6


def make_hires_subject(subdir, verbose=True):
    subdir = Path(subdir)
    subid = subdir.name
    subnum = re.search(r'(\d+)', subid).group(1)
    subdir = subdir / 'gifti'
    print(f'{time.ctime()}: Generating hi-res CIVET mesh for {subnum}')

    midthickness, sphere = (), ()
    for hemi in ('L', 'R'):
        fmt = f'{subid}_{{surf}}_surface_rsl_{HEMI[hemi]}_{{res}}.obj'

        # hi-res template sphere
        tpl = ATLASDIR / f'tpl-civet_den-164k_hemi-{hemi}_sphere.surf.gii'

        # lo-res native sphere
        proj = subdir / fmt.replace('.obj', '.surf.gii') \
                           .format(surf='sphere', res='81920')
        # low-res native midthickness
        fn = subdir.parent / 'surfaces' / fmt.format(surf='mid', res='81920')
        midthick = subdir / fn.name.replace('.obj', '.surf.gii')
        if not midthick.exists():
            images.obj_to_gifti(fn, midthick)
        # hi-res native midthickness (to-be-generated)
        midout = subdir / midthick.name.replace('81920', '327684')

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
        midthickness += (midout,)
        sphere += (msmout,)

    return midthickness, sphere


def main():
    subjects = np.loadtxt(DATADIR / 'subjects.txt', dtype=str)
    missing = np.loadtxt(DATADIR / 'missing.txt', dtype=str)
    subjects = np.core.defchararray.add(
        'HCP_', np.setdiff1d(subjects, missing)
    )

    midthicks, spheres = zip(*Parallel(n_jobs=N_PROC)(
        delayed(make_hires_subject)(CIVDIR / sub) for sub in subjects
    ))

    # calculate averaged hi-res spheres
    for surfs, hemi in zip(zip(*spheres), ('L', 'R')):
        sphere = images.average_surfaces(*surfs)
        fn = f'tpl-civet_space-fsLR_den-164k_hemi-{hemi}_sphere.surf.gii'
        nib.save(sphere, ATLASDIR / fn)
        run(SPHEREFIX.format(sphererot=ATLASDIR / fn), quiet=True)

    # also calculate midthickness average vertex areas (for use w/resampling)
    for mids, hemi in zip(zip(*midthicks), ('left', 'right')):
        vaavg = np.mean([images.vertex_areas(surf) for surf in mids], axis=0)
        fn = (f'tpl-civet_space-fsLR_den-164k_hemi-{hemi}_'
              'desc-vaavg_midthickness.surf.gii')
        nib.save(images.construct_shape_gii(vaavg), ATLASDIR / fn)


if __name__ == "__main__":
    main()
