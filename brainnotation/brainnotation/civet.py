# -*- coding: utf-8 -*-
"""
Contains code for generating CIVET <-> fsLR mappings (hopefully)
"""

import os
from pathlib import Path
from pkg_resources import resource_filename
import shutil
import tempfile

import nibabel as nib
import numpy as np

from netneurotools.civet import read_civet
from netneurotools.utils import run

from .points import get_shared_triangles, which_triangle

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
MSMCONF_ROT = resource_filename('brainnotation', 'data/MSMPreRotationConf')
MSMCONF_SUL = resource_filename('brainnotation', 'data/MSMSulcStrainFinalConf')
REFMESH = resource_filename('brainnotation', 'data/fsaverage.{hemi}_LR.'
                            'spherical_std.164k_fs_LR.surf.gii')
REFSULC = resource_filename('brainnotation', 'data/{hemi}.refsulc.164k_fs_LR.'
                            'shape.gii')
MSMPATH = resource_filename('brainnotation', 'data/msm')


def construct_surf_gii(vert, tri):
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


def construct_shape_gii(data):
    """
    Constructs shape gifti image from `data`

    Parameters
    ----------
    data : (N,) array_like
        Input data

    Returns
    -------
    img : nib.gifti.GiftiImage
        Shape image
    """

    return nib.GiftiImage(darrays=[
        nib.gifti.GiftiDataArray(data, intent='NIFTI_INTENT_SHAPE',
                                 datatype='NIFTI_TYPE_FLOAT32')
    ])


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

    img = construct_surf_gii(*read_civet(Path(obj)))
    if fn is None:
        fn = obj
    fn = Path(fn).resolve()
    if fn.name.endswith('.obj'):
        fn = fn.parent / fn.name.replace('.obj', '.surf.gii')
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

    img = construct_surf_gii(*nib.freesurfer.read_geometry(Path(surf)))
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
    img = construct_shape_gii(data)
    if fn is None:
        fn = morph + '.shape.gii'
    fn = Path(fn)
    nib.save(img, fn)

    return fn


def read_surfmap(surfmap):
    """
    Reads surface map from CIVET

    Parameters
    ----------
    surfmap : str or os.PathLike
        Surface mapping file to be loaded

    Returns
    -------
    control : (N,) array_like
        Control vertex IDs
    v0, v1 : (N,) array_like
        Target vertex IDs
    t : (N, 3) array_like
        Resampling weights
    """

    control, v0, v1, t1, t2 = np.loadtxt(surfmap, skiprows=4).T
    control = control.astype(int)
    v0 = v0.astype(int)
    v1 = v1.astype(int)
    t0 = 1 - t1 - t2

    return control, v0, v1, np.column_stack((t0, t1, t2))


def resample_surface_map(source, morph, target, surfmap):
    """
    Resamples `morph` data defined on `source` surface to `target` surface

    Uses `surfmap` to define mapping

    Inputs
    ------
    source : str or os.PathLike
        Path to surface file on which `morph` is defined
    morph : str or os.PathLike
        Path to morphology data defined on `source` surface
    target : str or os.PathLike
        Path to surface file on which to resample `morph` data
    surfmap : str or os.PathLike
        Path to surface mapping file defining transformation (CIVET style)

    Returns
    -------
    resampled : np.ndarray
        Provided `morph` data resampled to `target` surface
    """

    if isinstance(source, (str, os.PathLike)):
        source = read_civet(source)
    if isinstance(morph, (str, os.PathLike)):
        morph = np.loadtxt(morph)
    if len(morph) != len(source[0]):
        raise ValueError('Provided `morph` file has different number of '
                         'vertices from provided `source` surface')

    if isinstance(target, (str, os.PathLike)):
        target = read_civet(target)
    if isinstance(surfmap, (str, os.PathLike)):
        surfmap = read_surfmap(surfmap)
    if len(surfmap[0]) != len(target[0]):
        raise ValueError('Provided `target` surface has different number of '
                         'vertices from provided `surfmap` transformation.')

    source_tris = get_shared_triangles(source[1])
    resampled = np.zeros_like(morph)
    for (control, v0, v1, t) in zip(*surfmap):
        tris = source_tris[(v0, v1) if v0 < v1 else (v1, v0)]
        point, verts = target[0][control], source[0][tris]
        idx = which_triangle(point, verts)
        if idx is None:
            idx = np.argmin(np.linalg.norm(point - verts[:, -1], axis=1))
        resampled[control] = np.sum(morph[[v0, v1, tris[idx][-1]]] * t)

    return resampled


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

    generated = tuple()
    for n, hemi in enumerate(('left', 'right')):
        hemil = HEMI[hemi]
        params = dict(
            sphere=spheres[n],
            sulc=sulcs[n],
            msmpath=Path(MSMPATH),
            refmesh=Path(REFMESH.format(hemi=hemil)),
            refdata=Path(REFSULC.format(hemi=hemil)),
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
