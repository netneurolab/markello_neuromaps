# -*- coding: utf-8 -*-
"""
Contains code for generating CIVET <-> fsLR mappings (hopefully)
"""

from pathlib import Path
from pkg_resources import resource_filename
import shutil
import tempfile

import nibabel as nib
import numpy as np

from netneurotools.civet import read_civet
from netneurotools.utils import run

HEMI = dict(left='L', lh='L', L='L', right='R', rh='R', R='R')
SMOOTH = 'mris_smooth -n 3 -nw {white} {smoothwm}'
INFLATE = 'mris_inflate {smoothwm} {inflated}'
SPHERE = 'mris_sphere {inflated} {sphere}'
AFFREG = 'wb_command -surface-affine-regression {sphere} {trg} {affine}'
AFFAPP = 'wb_command -surface-apply-affine {sphere} {affine} {sphererot}'
SPHEREFIX = 'wb_command -surface-modify-sphere {sphererot} 100 {sphererot} ' \
            '-recenter'
MSMSULC = '{msmpath} {opts}--verbose --inmesh={sphere} --refmesh={refmesh} ' \
          '--indata={sulc} --refdata={refdata} --conf={conf} --out={outdir}'
MSMCONF_ROT = resource_filename('brainnotation', 'data/MSMPreRotationConf')
MSMCONF_SUL = resource_filename('brainnotation', 'data/MSMSulcStrainFinalConf')
REFMESH = resource_filename('brainnotation', 'data/fsaverage.{hemi}_LR.'
                            'spherical_std.164k_fs_LR.surf.gii')
REFSULC = resource_filename('brainnotation', 'data/{hemi}.refsulc.164k_fs_LR.'
                            'shape.gii')
MSMPATH = resource_filename('brainnotation', 'data/msm')


def _construct_gifti(vert, tri):
    """
    Constructs surface gifti image from `vert` and `tri`

    Parameters
    ----------
    vert : (N, 3)
        Vertices of surface mesh
    tri : (T, 3)
        Triangles comprising surface mesh

    Returns
    -------
    img : nib.gifti.GiftiImage
        Surface image
    """

    vert = nib.gifti.GiftiDataArray(vert, 'NIFTI_INTENT_POINTSET',
                                    'NIFTI_TYPE_FLOAT32',
                                    coordsys=nib.gifti.GiftiCoordSystem(3, 3))
    tri = nib.gifti.GiftiDataArray(tri, 'NIFTI_INTENT_TRIANGLE',
                                   'NIFTI_TYPE_INT32')
    img = nib.GiftiImage(darrays=[vert, tri])

    return img


def fix_coordsys(fn, val=3):
    """
    Sets {xform,data}space of coordsys for GIFTI image `fn` to `val`

    Parameters
    ----------
    fn : str or os.PathLike
        Path to GIFTI image

    Returns
    -------
    fn : os.PathLike
        Path to GIFTI image
    """

    fn = Path(fn)
    img = nib.load(fn)
    for attr in ('dataspace', 'xformspace'):
        setattr(img.darrays[0].coordsys, attr, val)
    nib.save(img, fn)
    return fn


def obj_to_gifti(obj, fn=None):
    """
    Converts CIVET `obj` surface file to GIFTI format

    Parameters
    ----------
    obj : str or os.PathLike
        CIVET file to be converted
    fn : str or os.PathLike, None
        Output filename. If not supplied uses input `obj` filename (with
        appropriate suffix). Default: None

    Returns
    -------
    fn : os.PathLike
        Path to saved image file
    """

    img = _construct_gifti(*read_civet(Path(obj)))
    if fn is None:
        fn = obj
    if fn.name.endswith('.obj'):
        fn = fn.parent / fn.name.replace('.obj', '.surf.gii')
    fn = Path(fn)
    nib.save(img, fn)

    return fn


def fssurf_to_gifti(surf, fn=None):
    """
    Converts FreeSurfer `surf` surface file to GIFTI format

    Parameters
    ----------
    obj : str or os.PathLike
        FreeSurfer surface file to be converted
    fn : str or os.PathLike, None
        Output filename. If not supplied uses input `surf` filename (with
        appropriate suffix). Default: None

    Returns
    -------
    fn : os.PathLike
        Path to saved image file
    """

    img = _construct_gifti(*nib.freesurfer.read_geometry(Path(surf)))
    if fn is None:
        fn = surf + '.surf.gii'
    fn = Path(fn)
    nib.save(img, fn)

    return fn


def fsmorph_to_gifti(morph, fn=None, modifier=None):
    """
    Converts FreeSurfer `morph` data file to GIFTI format

    Parameters
    ----------
    obj : str or os.PathLike
        FreeSurfer morph file to be converted
    fn : str or os.PathLike, None
        Output filename. If not supplied uses input `morph` filename (with
        appropriate suffix). Default: None
    modifier : float, optional
        Scalar factor to modify (multiply) the morphometric data. Default: None

    Returns
    -------
    fn : os.PathLike
        Path to saved image file
    """

    data = nib.freesurfer.read_morph_data(Path(morph))
    if modifier is not None:
        data *= float(modifier)
    img = nib.GiftiImage(darrays=[
        nib.gifti.GiftiDataArray(data, intent='NIFTI_INTENT_SHAPE',
                                 datatype='NIFTI_TYPE_FLOAT32')
    ])
    if fn is None:
        fn = morph + '.shape.gii'
    fn = Path(fn)
    nib.save(img, fn)

    return fn


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


def register_subject(subjdir, affine=None, only_gen_affine=True):
    """
    Registers CIVET processed `subjdir` to fsLR spacae

    Parameters
    ----------
    subjdir : str or os.PathLike
        Path to CIVET output subject directory
    affine : (2,) tuple-of-str or os.PathLike, optional
        Filepaths to affine (rotation) transform matrix to rotate data from
        `subjdir` into appropriate output space. Tuple should be (left, right)
        matrices. If not provided will be generated for the subject using MSM.
        Default: None
    only_gen_affine : bool, optional
        Whether to stop the registration procedure after the initial rotation
        affine matrix is generated. Useful if you want to e.g., average
        rotation matrices across subjects. Default: False

    Returns
    -------
    generated : (2,) tuple-of-os.PathLike
        Subject spheres (left, right hemispheres) aligned to fsLR space. If
        `only_gen_affine` is True then these are the affine rotation matrices
        instead of the surface meshes.ls
    """

    subjdir = Path(subjdir).resolve()
    prefix = subjdir.name
    suffix = 'surface_rsl_{hemi}_81920.{suff}'
    fmt = subjdir / 'surfaces' / f'{prefix}_{{surf}}_{suffix}'

    # set up working directory
    tempdir = Path(tempfile.gettempdir()) / prefix
    tempdir.mkdir(exist_ok=True, parents=True)

    generated = tuple()
    for n, hemi in enumerate(('left', 'right')):
        white = Path(str(fmt).format(surf='white', hemi=hemi, suff='obj'))
        white = obj_to_gifti(white, fn=tempdir / white.name)
        hfmt = fmt.name.format(surf='{surf}', hemi=hemi, suff='surf.gii')
        params = dict(
            tempdir=tempdir,
            white=white,
            smoothwm=tempdir / 'rh.smoothwm',
            inflated=tempdir / 'rh.inflated',
            sulcout=tempdir / 'rh.sulc',
            sphere=tempdir / hfmt.format(surf='sphere'),
            sulc=tempdir / fmt.name.format(surf='sulc', hemi=hemi,
                                           suff='shape.gii'),
            msmpath=Path(MSMPATH),
            refmesh=Path(REFMESH.format(hemi=HEMI[hemi])),
            refdata=Path(REFSULC.format(hemi=HEMI[hemi])),
            outdir=tempdir / 'msmrot' / f'{HEMI[hemi]}.',
            conf=Path(MSMCONF_ROT),
            trg=tempdir / 'msmrot' / f'{HEMI[hemi]}.sphere.reg.surf.gii',
            affine=tempdir / f'{prefix}_{hemi}_affine.txt',
            sphererot=tempdir / hfmt.format(surf='sphere_rot')
        )

        params['outdir'].parent.mkdir(exist_ok=True, parents=True)
        if not params['sphere'].exists() or not params['sulc'].exists():
            # generate a spherical representation of the data
            for func in (SMOOTH, INFLATE, SPHERE):
                run(func.format(**params))
            # fix the coordinate system on the generated sphere
            fix_coordsys(params['sphere'])
            # invert sulcal morphology to match HCP
            fsmorph_to_gifti(params['sulcout'], params['sulc'], -1)
            # get rid of intermediate files
            for fn in ('smoothwm', 'inflated', 'sulcout'):
                params[fn].unlink()
        if not params['affine'].exists():
            # run the pre-rotation MSM and generate the rotational affine
            # matrix to align the subject sphere to the HCP group sphere
            for func in (MSMSULC, AFFREG):
                run(func.format(**params, opts='--levels=2 '))
            rotation = extract_rotation(params['affine'])
            np.savetxt(params['affine'], rotation, fmt='%.10f')

        if only_gen_affine:
            generated += (params['affine'],)
            continue

        # if we provided a rotational affine matrix use that instead of what
        # we generated!
        if affine is not None:
            params['affine'] = affine[n]

        # apply rotation matrix to sphere surface
        if not params['sphererot'].exists():
            for func in (AFFAPP, SPHEREFIX):
                run(func.format(**params))

        # set up parameters for "final" MSM (rotated sphere -> HCP group)
        params['sphere'] = params['sphererot']
        params['conf'] = MSMCONF_SUL
        params['outdir'] = tempdir / 'msmsulc' / f'{HEMI[hemi]}.'
        params['outdir'].parent.mkdir(exist_ok=True, parents=True)
        output = tempdir / hfmt.format(surf='sphere_final')
        if not output.exists():
            run(MSMSULC.format(**params, opts=''))
            fn = params['outdir'].parent / f'{HEMI[hemi]}.sphere.reg.surf.gii'
            shutil.copy(fn, output)

        generated += (output,)

    return generated
