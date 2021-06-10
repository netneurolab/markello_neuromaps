# -*- coding: utf-8 -*-
"""
Contains code for generating CIVET <-> fsLR mappings (hopefully)
"""

from pathlib import Path
from pkg_resources import resource_filename
import shutil
import tempfile

import numpy as np

from brainnotation.datasets import fetch_atlas
from brainnotation.images import (fix_coordsys, obj_to_gifti, fsmorph_to_gifti)
from brainnotation.utils import run

HEMI = dict(left='L', lh='L', L='L', right='R', rh='R', R='R')
SMOOTH = 'mris_smooth -n 3 -nw {white} {smoothwm}'
INFLATE = 'mris_inflate {smoothwm} {inflated}'
SPHERE = 'mris_sphere {inflated} {sphere}'
AFFREG = 'wb_command -surface-affine-regression {sphere} {trg} {affine}'
AFFAPP = 'wb_command -surface-apply-affine {sphere} {affine} {sphererot}'
SPHEREFIX = 'wb_command -surface-modify-sphere {sphererot} 100 {sphererot} ' \
            '-recenter'
MSMROT = '{msmpath} --levels=2 --verbose --inmesh={sphere} --indata={sulc} ' \
         '--refmesh={refmesh} --refdata={refdata} --conf={rotconf} ' \
         '--out={msmrotout}'
MSMSUL = '{msmpath} --verbose --inmesh={sphererot} --indata={sulc} ' \
         '--refmesh={refmesh} --refdata={refdata} --conf={sulconf} '\
         '--out={msmsulout}'
MSMPATH = resource_filename('brainnotation_dev', 'data/msm')
MSMCONF_ROT = resource_filename('brainnotation_dev',
                                'data/MSMPreRotationConf')
MSMCONF_SUL = resource_filename('brainnotation_dev',
                                'data/MSMSulcStrainFinalConf')


def extract_rotation(affine):
    """
    Extracts rotational component from `affine` matrix

    Parameters
    ----------
    affine : str or (4, 4) array_like
        Input affine matrix

    Returns
    -------
    rotation : (4, 4) np.ndarray
        Rotation matrix

    References
    -----------
    https://github.com/ecr05/dHCP_template_alignment
    """

    try:
        affine = np.loadtxt(affine)
    except TypeError:
        pass

    if affine.shape != (4, 4):
        raise ValueError('Provided affine matrix must be of shape (4, 4)')

    u, _, v = np.linalg.svd(affine[:3, :3])
    out = np.zeros_like(affine)
    out[:3, :3] = u @ v
    out[3, 3] = 1

    return out


def register_subject(subdir, affine=None, only_gen_affine=True, verbose=False):
    """
    Registers CIVET processed `subdir` to fsLR spacae

    Parameters
    ----------
    subdir : str or os.PathLike
        Path to CIVET output subject directory
    affine : (2,) tuple-of-str or os.PathLike, optional
        Filepaths to affine (rotation) transform matrix to rotate data from
        `subdir` into appropriate output space. Tuple should be (left, right)
        matrices. If not provided will be generated for the subject using MSM.
        Default: None
    only_gen_affine : bool, optional
        Whether to stop the registration procedure after the initial rotation
        affine matrix is generated. Useful if you want to e.g., average
        rotation matrices across subjects. Default: False
    verbose : bool, optional
        Whether to print status messages. Default: False

    Returns
    -------
    generated : (2,) tuple-of-os.PathLike
        Subject spheres (left, right hemispheres) aligned to fsLR space. If
        `only_gen_affine` is True then these are the affine rotation matrices
        instead of the surface meshes.
    """

    if affine is not None:
        if len(affine) != 2:
            raise ValueError('If providing `affine` it must be len-2 tuple')

    subdir = Path(subdir).resolve()
    prefix = subdir.name
    fmt = f'{prefix}_{{surf}}_surface_rsl_{{hemi}}_81920.{{suff}}'

    # set up working directory
    tempdir = Path(tempfile.gettempdir()) / prefix
    tempdir.mkdir(exist_ok=True, parents=True)

    # we need the spherical meshes + sulcal depth maps for the subjects
    spheres, sulcs = civet_sphere(subdir, resampled=True)

    # get reference fsLR mesh + sulcal depth info
    fslr = fetch_atlas('fslr', '164k', verbose=verbose)

    generated = tuple()
    for n, hemi in enumerate(('left', 'right')):
        hemil = HEMI[hemi]
        params = dict(
            sphere=spheres[n],
            sulc=sulcs[n],
            msmpath=Path(MSMPATH),
            refmesh=getattr(fslr['sphere'], hemil),
            refdata=getattr(fslr['sulc'], hemil),
            msmrotout=tempdir / 'msmrot' / f'{hemil}.',
            msmsulout=tempdir / 'msmsulc' / f'{hemil}.',
            rotconf=Path(MSMCONF_ROT),
            sulconf=Path(MSMCONF_SUL),
            trg=tempdir / 'msmrot' / f'{hemil}.sphere.reg.surf.gii',
            affine=tempdir / f'{prefix}_{hemi}_affine.txt',
            sphererot=tempdir / fmt.format(surf='sphere_rot', hemi=hemi,
                                           suff='surf.gii')
        )
        for key in ('msmrotout', 'msmsulout'):
            params[key].parent.mkdir(exist_ok=True, parents=True)

        # run the pre-rotation MSM and generate the rotational affine matrix to
        # align the subject sphere to the fsLR rotated sphere. if we provided a
        # rotational affine matrix use that instead
        if not params['affine'].exists() and affine is None:
            for func in (MSMROT, AFFREG):
                run(func.format(**params), quiet=not verbose)
            rotation = extract_rotation(params['affine'])
            np.savetxt(params['affine'], rotation, fmt='%.10f')
        elif affine is not None:
            params['affine'] = affine[n]

        # if we only want to generate first-pass affines then abort here
        if only_gen_affine:
            generated += (params['affine'],)
            continue

        # apply rotation matrix to sphere surface
        if not params['sphererot'].exists():
            for func in (AFFAPP, SPHEREFIX):
                run(func.format(**params), quiet=not verbose)

        # set up parameters for "final" MSM (rotated sphere -> HCP group)
        output = tempdir / fmt.format(surf='sphere_final', hemi=hemi,
                                      suff='surf.gii')
        if not output.exists():
            run(MSMSUL.format(**params), quiet=not verbose)
            fn = str(params['msmsulout']) + 'sphere.reg.surf.gii'
            shutil.copy(fn, output)

        generated += (output,)

    return generated


def civet_sphere(subdir, resampled=True, verbose=True):
    """
    Generates spherical surface mesh for CIVET-processed `subdir`

    Parameters
    ----------
    subdir : str or os.PathLike
        Path to CIVET output subject directory
    resampled : bool, optional
        Whether to generate spheres of resampled instead of native surfaces.
        Default: True
    verbose : bool, optional
        Whether to print status messages. Default: True

    Returns
    -------
    sphere : (2,) tuple-of-os.PathLike
        Path to generated sphere surface meshes for (left, right) hemisphere
    sulc : (2,) tuple-of-os.PathLike
        Path to generated sulcal depth maps for (left, right) hemisphere
    """

    subdir = Path(subdir).resolve()
    prefix = subdir.name
    rsl = 'rsl_' if resampled else ''
    suffix = '{hemi}_81920.{suff}'
    fmt = subdir / 'surfaces' / f'{prefix}_{{surf}}_surface_{rsl}{suffix}'

    # set up working directory
    tempdir = Path(tempfile.gettempdir()) / (prefix + '_sphere')
    tempdir.mkdir(exist_ok=True, parents=True)

    sphere, sulc = tuple(), tuple()
    for hemi in ('left', 'right'):
        white = Path(str(fmt).format(surf='white', hemi=hemi, suff='obj'))
        params = dict(
            white=obj_to_gifti(white, fn=tempdir / white.name),
            smoothwm=tempdir / f'{hemi[0]}h.smoothwm',
            inflated=tempdir / f'{hemi[0]}h.inflated',
            sulcout=tempdir / f'{hemi[0]}h.sulc',
            sphere=subdir / 'gifti' / fmt.name.format(surf='sphere', hemi=hemi,
                                                      suff='surf.gii'),
            sulc=subdir / 'gifti' / fmt.name.format(surf='sulc', hemi=hemi,
                                                    suff='shape.gii')
        )

        # generate spherical representation of the CIVET data; we'll fix
        # the coordinate system on the generated sphere + invert sulcal
        # morphology (* -1) to match HCP format
        if not params['sphere'].exists():
            params['sphere'].parent.mkdir(exist_ok=True, parents=True)
            for func in (SMOOTH, INFLATE, SPHERE):
                run(func.format(**params), quiet=not verbose)
            fix_coordsys(params['sphere'])
            fsmorph_to_gifti(params['sulcout'], params['sulc'], -1)
            for fn in ('white', 'smoothwm', 'inflated', 'sulcout'):
                params[fn].unlink()

        sphere += (params['sphere'],)
        sulc += (params['sulc'],)

    shutil.rmtree(tempdir)

    return sphere, sulc
