# -*- coding: utf-8 -*-
"""
Contains code for generating registration-fusion mappings
"""

from collections import defaultdict
from pathlib import Path
from nibabel.filebasedimages import ImageFileError
from pkg_resources import resource_filename
import tempfile

import nibabel as nib
import numpy as np

from neuromaps.civet import resample_surface_map
from neuromaps.datasets import fetch_atlas
from neuromaps.images import (construct_shape_gii, obj_to_gifti,
                                  minc_to_nifti)
from neuromaps.utils import tmpname, run, check_fs_subjid


VOLTOSURF = 'wb_command -volume-to-surface-mapping {volume} {srcmid} ' \
            '{out} -ribbon-constrained {white} {pial} -interpolate TRILINEAR'
SURFTOVOL = 'wb_command -metric-to-volume-mapping {out} {trgmid} {t1w} ' \
            '{volume} -ribbon-constrained {white} {pial} -greedy'
NATTOTEMP = 'wb_command -metric-resample {out} {src} {trg} ADAP_BARY_AREA ' \
            '{resamp} -area-surfs {srcmid} {trgmid} -current-roi {natmask}'
TEMPTONAT = 'wb_command -metric-resample {resamp} {src} {trg} ADAP_BARY_AREA' \
            ' {out} -area-surfs {srcmid} {trgmid} -current-roi {fsmask}'
NATMASK = 'wb_command -metric-mask {out} {natmask} {out}'
FSMASK = 'wb_command -metric-mask {resamp} {fsmask} {resamp}'
MNITOT1W = 'wb_command -volume-resample {mni} {space} CUBIC {volume} ' \
           '-warp {warp} -fnirt {space}'
T1WTOMNI = 'wb_command -volume-resample {volume} {space} ENCLOSING_VOXEL ' \
           '{mni} -warp {warp} -fnirt {space}'
GENMIDTHICK = 'wb_command -surface-average {mid} -surf {white} -surf {pial}'
FSTOGII = 'mris_convert {fs} {gii}'


def make_xyz(template):
    """
    Makes x/y/z index images in space of `template`

    Parameters
    ----------
    template : str
        Filepath to template

    Returns
    -------
    x, y, z : nib.Nifti1Image
        Index images in space of template
    """
    img = nib.load(template)
    ijk = np.meshgrid(*[range(d) for d in img.shape])
    xyz = nib.affines.apply_affine(img.affine, np.stack(ijk, axis=-1))
    x, y, z = [
        img.__class__(data.squeeze().transpose(1, 0, 2), img.affine)
        for data in np.split(xyz, 3, axis=-1)
    ]
    return x, y, z


def make_surf_xyz(template):
    """
    Makes x/y/z index images in space of surface `template`

    Parameters
    ----------
    template : str
        Filepath to surface template

    Returns
    -------
    x, y, z : nib.GiftImage
        Index images in space of template
    """

    try:
        xyz = nib.load(template).agg_data('NIFTI_INTENT_POINTSET')
    except ImageFileError:
        xyz = nib.freesurfer.read_geometry(template)[0]
    x, y, z = [construct_shape_gii(data) for data in np.split(xyz, 3, axis=-1)]
    return x, y, z


def get_files(subject, out_dir=None, profile='hcp', verbose=False):
    """
    Gets necessary reg-fusion input files for HCP `subject`

    Parameters
    ----------
    subject : str
        HCP subject ID
    out_dir : str or os.PathLike
        Path where data should be downloaded
    profile : str, optional
        AWS profile name to use for connection. Default: 'hcp'
    verbose : bool, optional
        Whether to print status updates while downloading files. Default: False

    Returns
    -------
    out_dir : pathlib.Path
        Path to downloaded data
    """

    import boto3
    from botocore.exceptions import ClientError

    fn = resource_filename('neuromaps_dev', 'data/hcpfiles.txt')
    with open(fn) as src:
        fnames = [fn.strip().format(sub=subject) for fn in src.readlines()]

    if out_dir is None:
        out_dir = tempfile.gettempdir()
    out_dir = Path(out_dir) / subject
    out_dir.mkdir(exist_ok=True, parents=True)

    # open session with specified profile
    s3 = boto3.Session(profile_name=profile).client('s3')

    for fn in fnames:
        out_path = out_dir / fn
        out_path.parent.mkdir(exist_ok=True, parents=True)
        if not out_path.exists():
            if verbose:
                print(f'Downloading {out_path}')
            try:
                s3.download_file('hcp-openaccess',
                                 f'HCP_1200/{subject}/{fn}',
                                 str(out_path))
            except ClientError:
                print(f'Failed to download {out_path}')

    return out_dir


def hcp_regfusion(subdir, res='32k', verbose=True):
    """
    Runs registration fusion pipeline on HCP subject `subdir`

    Generates mapping from MNI --> fsLR space

    Parameters
    ----------
    subdir : str or os.PathLike
        Path to HCP subject directory
    res : {'32k', '164k'}, optional
        Resolution at which to project the data
    verbose : bool, optional
        Whether to print status messages. Default: False

    Returns
    -------
    generated : dict-of-list
        Dictionary with keys ('x', 'y', 'z') containing lists of left-/right-
        hemisphere projected index maps for registration fusion
    """

    # path handling
    resolutions = ('32k', '164k')
    if res not in resolutions:
        raise ValueError(f'Provided res must be one of {resolutions}. '
                         f'Received {res}')
    res = f'{res}_fs_LR'
    subdir = Path(subdir).resolve()
    subid = subdir.name
    t1wdir = subdir / 'T1w'
    subdir = subdir / 'MNINonLinear'
    if not subdir.exists():
        raise ValueError('Provided subdir does not have expected structure')
    resdir = subdir / ('fsaverage_LR32k' if res == '32k_fs_LR' else '')
    natdir = subdir / 'Native'

    # run the actual commands
    generated = defaultdict(list)
    for img, name in zip(make_xyz(subdir / 'T1w.nii.gz'), ('x', 'y', 'z')):
        template = tmpname(suffix='.nii.gz')
        nib.save(img, template)
        # transform index images from MNI to native space
        volume = tmpname(suffix=f'.{name}.nat.nii.gz')
        run(MNITOT1W.format(mni=template,
                            space=t1wdir / 'T1w_acpc_dc.nii.gz',
                            volume=volume,
                            warp=subdir / 'xfms' / 'standard2acpc_dc.nii.gz'))
        template.unlink()
        for hemi in ('L', 'R'):
            prefix = f'{subid}.{hemi}'
            params = dict(
                volume=volume,
                white=natdir / f'{prefix}.white.native.surf.gii',
                pial=natdir / f'{prefix}.pial.native.surf.gii',
                src=natdir / f'{prefix}.sphere.MSMAll.native.surf.gii',
                trg=resdir / f'{prefix}.sphere.{res}.surf.gii',
                srcmid=natdir / f'{prefix}.midthickness.native.surf.gii',
                trgmid=resdir / f'{prefix}.midthickness.{res}.surf.gii',
                natmask=natdir / f'{prefix}.roi.native.shape.gii',
                fsmask=resdir / f'{prefix}.atlasroi.{res}.shape.gii',
                out=tmpname(suffix='.func.gii'),
                resamp=tmpname(prefix=f'{hemi}.', suffix=f'.{name}.func.gii')
            )

            # now generate the reg-fusion outputs
            if verbose:
                print(f'Generating {name}.{hemi} index image: '
                      f'{params["resamp"]}')
            for cmd in (VOLTOSURF, NATMASK, NATTOTEMP, FSMASK):
                run(cmd.format(**params))
            generated[name].append(params['resamp'])
            params['out'].unlink()
        volume.unlink()

    return generated


def fs_regfusion(subdir, res='fsaverage6', verbose=False):
    """
    Runs registration fusion pipeline on HCP subject `subdir`

    Generates mapping from MNI --> fsaverage space

    Parameters
    ----------
    subdir : str or os.PathLike
        Path to HCP subject directory
    res : {'fsaverage', 'fsaverage4', 'fsaverage5', 'fsaverage6'}, optional
        Resolution at which to project the data
    verbose : bool, optional
        Whether to print status messages. Default: False

    Returns
    -------
    generated : dict-of-list
        Dictionary with keys ('x', 'y', 'z') containing lists of left-/right-
        hemisphere projected index maps for registration fusion
    """

    # some light path handling
    resolutions = (
        'fsaverage', 'fsaverage3', 'fsaverage4', 'fsaverage5', 'fsaverage6'
    )
    if res not in resolutions:
        raise ValueError(f'Provided res must be one of {resolutions}. '
                         f'Received {res}')
    subdir = Path(subdir).resolve()
    subid = subdir.name
    mnidir = subdir / 'MNINonLinear'
    t1wdir = subdir / 'T1w'

    res, fsdir = check_fs_subjid(res)
    fsdir = Path(fsdir) / res

    # run the actual commands
    generated = defaultdict(list)
    for img, name in zip(make_xyz(mnidir / 'T1w.nii.gz'), ('x', 'y', 'z')):
        # transform index images from MNI to native space
        template = tmpname(suffix='.nii.gz')
        nib.save(img, template)
        volume = tmpname(suffix=f'.{name}.nat.nii.gz')
        run(MNITOT1W.format(mni=template,
                            space=t1wdir / 'T1w_acpc_dc.nii.gz',
                            volume=volume,
                            warp=mnidir / 'xfms' / 'standard2acpc_dc.nii.gz'))
        template.unlink()
        for hemi in ('lh', 'rh'):
            params = dict(
                volume=volume,
                out=tmpname(prefix=f'{hemi}.', suffix=f'.{name}.nat.func.gii'),
                white=t1wdir / subid / 'surf' / f'{hemi}.white.surf.gii',
                pial=t1wdir / subid / 'surf' / f'{hemi}.pial.surf.gii',
                fswhite=fsdir / 'surf' / f'{hemi}.white',
                fspial=fsdir / 'surf' / f'{hemi}.pial',
                src=t1wdir / subid / 'surf' / f'{hemi}.sphere.reg',
                trg=fsdir / 'surf' / f'{hemi}.sphere',
                srcmid=('white', 'pial'),
                trgmid=('fswhite', 'fspial'),
                resamp=tmpname(prefix=f'{hemi}.', suffix=f'.{name}.func.gii'),
                natmask=t1wdir / subid / 'label' / f'{hemi}.cortex.label',
                fsmask=fsdir / 'label' / f'{hemi}.cortex.label',
            )

            # convert FS to gii for volume-to-surface-mapping
            for key in ('fswhite', 'fspial', 'src', 'trg'):
                fs = params[key]
                gii = tmpname(prefix=f'{hemi}.',
                              suffix=f'.{name}.{key}.surf.gii')
                run(FSTOGII.format(fs=fs, gii=gii), quiet=True)
                params[key] = gii

            # generate midthickness files
            for key in ('srcmid', 'trgmid'):
                white, pial = [params[k] for k in params[key]]
                gii = tmpname(prefix=f'{hemi}.',
                              suffix=f'.{name}.{key}.surf.gii')
                run(GENMIDTHICK.format(mid=gii, white=white, pial=pial))
                params[key] = gii

            # generate medial mask files
            for key in ('natmask', 'fsmask'):
                surf = params['src'] if key == 'natmask' else params['trg']
                n_vert = len(nib.load(surf).agg_data('NIFTI_INTENT_POINTSET'))
                mask = np.zeros(n_vert, dtype='int32')
                mask[nib.freesurfer.read_label(params[key])] = 1
                gii = tmpname(prefix=f'{hemi}.',
                              suffix=f'.{name}.{key}.label.gii')
                nib.save(nib.GiftiImage(darrays=[
                    nib.gifti.GiftiDataArray(mask,
                                             intent='NIFTI_INTENT_LABEL',
                                             datatype='NIFTI_TYPE_INT32')
                ]), gii)
                params[key] = gii

            # now generate the reg-fusion mapping (subject -> fsaverage)
            if verbose:
                print(f'Generating {name}.{hemi} index image: '
                      f'{params["resamp"]}')
            for cmd in (VOLTOSURF, NATMASK, NATTOTEMP, FSMASK):
                run(cmd.format(**params))
            generated[name].append(params['resamp'])
            for key in ('out', 'fswhite', 'fspial', 'src', 'trg',
                        'srcmid', 'trgmid', 'natmask', 'fsmask'):
                params[key].unlink()
        volume.unlink()

    return generated


def civet_regfusion(subdir, res='41k', verbose=False):
    """
    Runs registration fusion pipeline on HCP subject `subdir`

    Generates mapping from MNI --> CIVET space

    Parameters
    ----------
    subdir : str or os.PathLike
        Path to HCP subject directory
    res : {'41k'}, optional
        Resolution at which to project the data
    verbose : bool, optional
        Whether to print status messages. Default: False

    Returns
    -------
    generated : dict-of-list
        Dictionary with keys ('x', 'y', 'z') containing lists of left-/right-
        hemisphere projected index maps for registration fusion
    """

    # path handling
    subdir = Path(subdir).resolve()
    subid = subdir.name
    surfdir = subdir / 'surfaces'
    natobj = f'{subid}_{{surf}}_surface_{{hemi}}_81920.obj'
    rslobj = f'{subid}_{{surf}}_surface_rsl_{{hemi}}_81920.obj'
    mnc = subdir / 'final' / f'{subid}_t1_tal.mnc'

    tempdir = Path(tempfile.gettempdir()) / (subid + '_regfusion')
    tempdir.mkdir(exist_ok=True, parents=True)

    # civet medial wall
    civet_medial = fetch_atlas('civet', res)['medial']

    # run the actual commands
    generated = defaultdict(list)
    nii = minc_to_nifti(mnc, mnc)
    for img, name in zip(make_xyz(nii), ('x', 'y', 'z')):
        template = tmpname(suffix='.nii.gz', directory=tempdir)
        nib.save(img, template)
        for hemi, medial in zip(('left', 'right'), civet_medial):
            params = dict(
                volume=template,
                white=surfdir / natobj.format(surf='white', hemi=hemi),
                pial=surfdir / natobj.format(surf='gray', hemi=hemi),
                srcmid=surfdir / natobj.format(surf='mid', hemi=hemi),
                trgmid=surfdir / rslobj.format(surf='mid', hemi=hemi),
                fsmask=medial,
                surfmap=(subdir / 'transforms' / 'surfreg'
                         / f'{subid}_{hemi}_surfmap.sm'),
                out=tmpname(suffix='.func.gii', directory=tempdir),
                resamp=tmpname(prefix=f'{hemi}.', suffix=f'.{name}.func.gii',
                               directory=tempdir)
            )

            # we need gifti images!
            for key in ('white', 'pial', 'srcmid', 'trgmid'):
                fn = tempdir / params[key].name
                params[key] = obj_to_gifti(params[key], fn)

            # now generate the reg-fusion outputs
            if verbose:
                print(f'Generating {name}.{hemi} index image: '
                      f'{params["resamp"]}')

            # project data to the surface
            run(VOLTOSURF.format(**params))

            # resample the data to the ICBM surface
            source = nib.load(params['srcmid']).agg_data()
            morph = nib.load(params['out']).agg_data()
            target = nib.load(params['trgmid']).agg_data()
            resamp = resample_surface_map(source, morph, target,
                                          params['surfmap'])
            nib.save(construct_shape_gii(resamp), params['resamp'])

            # mask the resampled data
            run(FSMASK.format(**params))

            generated[name].append(params['resamp'])
            for key in ('white', 'pial', 'srcmid', 'trgmid', 'out'):
                params[key].unlink()

        template.unlink()

    return generated


def group_regfusion(x, y, z):
    """
    Generates group-level registration fusion mappings

    Parameters
    ----------
    x, y, z : list of images
        List of reg-fusion output images to be averaged across subjects

    Returns
    -------
    mappings : np.ndarray
        Registration fusion mapping coordinates
    """

    out = ()
    for imgs in (x, y, z):
        sniff = np.zeros(nib.load(imgs[0]).agg_data().shape)
        for img in imgs:
            sniff += nib.load(img).agg_data()
        sniff /= len(imgs)
        out += (sniff,)

    return np.column_stack(out)


def hcp_surf_regfusion(subdir, res='32k', verbose=True):
    """
    Runs registration fusion pipeline on HCP subject `subdir`

    Generates mapping from fsLR --> MNI space

    Parameters
    ----------
    subdir : str or os.PathLike
        Path to HCP subject directory
    res : {'32k', '164k'}, optional
        Resolution from which to project the data
    verbose : bool, optional
        Whether to print status messages. Default: False

    Returns
    -------
    generated : dict-of-list
        Dictionary with keys ('x', 'y', 'z') containing lists of left-/right-
        hemisphere projected index maps for registration fusion
    """

    # path handling
    resolutions = ('32k', '164k')
    if res not in resolutions:
        raise ValueError(f'Provided res must be one of {resolutions}. '
                         f'Received {res}')
    res = f'{res}_fs_LR'
    subdir = Path(subdir).resolve()
    subid = subdir.name
    t1wdir = subdir / 'T1w'
    subdir = subdir / 'MNINonLinear'
    if not subdir.exists():
        raise ValueError('Provided subdir does not have expected structure')
    resdir = subdir / ('fsaverage_LR32k' if res == '32k_fs_LR' else '')
    natdir = subdir / 'Native'

    # run the actual commands
    generated = defaultdict(list)
    for hemi in ('L', 'R'):
        prefix = f'{subid}.{hemi}'
        temp = resdir / f'{prefix}.sphere.{res}.surf.gii'
        for img, name in zip(make_surf_xyz(temp), ('x', 'y', 'z')):
            template = tmpname(suffix='.shape.gii')
            nib.save(img, template)

            params = dict(
                resamp=template,
                volume=tmpname(prefix=f'{hemi}.', suffix=f'{name}.nat.nii.gz'),
                mni=tmpname(prefix=f'{hemi}.', suffix=f'.{name}.nii.gz'),
                space=subdir / 'T1w.nii.gz',
                t1w=t1wdir / 'T1w_acpc_dc.nii.gz',
                warp=subdir / 'xfms' / 'acpc_dc2standard.nii.gz',
                white=natdir / f'{prefix}.white.native.surf.gii',
                pial=natdir / f'{prefix}.pial.native.surf.gii',
                trg=natdir / f'{prefix}.sphere.MSMAll.native.surf.gii',
                src=resdir / f'{prefix}.sphere.{res}.surf.gii',
                trgmid=natdir / f'{prefix}.midthickness.native.surf.gii',
                srcmid=resdir / f'{prefix}.midthickness.{res}.surf.gii',
                natmask=natdir / f'{prefix}.roi.native.shape.gii',
                fsmask=resdir / f'{prefix}.atlasroi.{res}.shape.gii',
                out=tmpname(suffix='.func.gii'),
            )

            # now generate the reg-fusion outputs
            if verbose:
                print(f'Generating {name}.{hemi} index image: ', params['mni'])
            for cmd in (FSMASK, TEMPTONAT, NATMASK, SURFTOVOL, T1WTOMNI):
                run(cmd.format(**params))
            generated[name].append(params['mni'])
            for key in ('resamp', 'volume'):
                params[key].unlink()

    return generated


def fs_surf_regfusion(subdir, res='fsaverage6', verbose=False):
    """
    Runs registration fusion pipeline on HCP subject `subdir`

    Generates mapping from fsaverage --> MNI space

    Parameters
    ----------
    subdir : str or os.PathLike
        Path to HCP subject directory
    res : {'fsaverage', 'fsaverage4', 'fsaverage5', 'fsaverage6'}, optional
        Resolution from which to project the data
    verbose : bool, optional
        Whether to print status messages. Default: False

    Returns
    -------
    generated : dict-of-list
        Dictionary with keys ('x', 'y', 'z') containing lists of left-/right-
        hemisphere projected index maps for registration fusion
    """
    resolutions = (
        'fsaverage', 'fsaverage3', 'fsaverage4', 'fsaverage5', 'fsaverage6'
    )
    if res not in resolutions:
        raise ValueError(f'Provided res must be one of {resolutions}. '
                         f'Received {res}')
    subdir = Path(subdir).resolve()
    subid = subdir.name
    mnidir = subdir / 'MNINonLinear'
    t1wdir = subdir / 'T1w'

    res, fsdir = check_fs_subjid(res)
    fsdir = Path(fsdir) / res

    # run the actual commands
    generated = defaultdict(list)
    for hemi in ('lh', 'rh'):
        for img, name in zip(make_surf_xyz(fsdir / 'surf' / f'{hemi}.sphere'),
                             ('x', 'y', 'z')):
            template = tmpname(suffix='.shape.gii')
            nib.save(img, template)

            params = dict(
                resamp=template,
                volume=tmpname(prefix=f'{hemi}.', suffix=f'{name}.nat.nii.gz'),
                mni=tmpname(prefix=f'{hemi}.', suffix=f'.{name}.nii.gz'),
                space=mnidir / 'T1w.nii.gz',
                t1w=t1wdir / 'T1w_acpc_dc.nii.gz',
                warp=mnidir / 'xfms' / 'acpc_dc2standard.nii.gz',
                out=tmpname(prefix=f'{hemi}.', suffix=f'.{name}.nat.func.gii'),
                white=t1wdir / subid / 'surf' / f'{hemi}.white.surf.gii',
                pial=t1wdir / subid / 'surf' / f'{hemi}.pial.surf.gii',
                fswhite=fsdir / 'surf' / f'{hemi}.white',
                fspial=fsdir / 'surf' / f'{hemi}.pial',
                trg=t1wdir / subid / 'surf' / f'{hemi}.sphere.reg',
                src=fsdir / 'surf' / f'{hemi}.sphere',
                trgmid=('white', 'pial'),
                srcmid=('fswhite', 'fspial'),
                natmask=t1wdir / subid / 'label' / f'{hemi}.cortex.label',
                fsmask=fsdir / 'label' / f'{hemi}.cortex.label',
            )

            # convert FS to gii for volume-to-surface-mapping
            for key in ('fswhite', 'fspial', 'src', 'trg'):
                fs = params[key]
                gii = tmpname(prefix=f'{hemi}.',
                              suffix=f'.{name}.{key}.surf.gii')
                run(FSTOGII.format(fs=fs, gii=gii), quiet=True)
                params[key] = gii

            # generate midthickness files
            for key in ('srcmid', 'trgmid'):
                white, pial = [params[k] for k in params[key]]
                gii = tmpname(prefix=f'{hemi}.',
                              suffix=f'.{name}.{key}.surf.gii')
                run(GENMIDTHICK.format(mid=gii, white=white, pial=pial))
                params[key] = gii

            # generate medial mask files
            for key in ('natmask', 'fsmask'):
                surf = params['trg'] if key == 'natmask' else params['src']
                n_vert = len(nib.load(surf).agg_data('NIFTI_INTENT_POINTSET'))
                mask = np.zeros(n_vert, dtype='int32')
                mask[nib.freesurfer.read_label(params[key])] = 1
                gii = tmpname(prefix=f'{hemi}.',
                              suffix=f'.{name}.{key}.label.gii')
                nib.save(nib.GiftiImage(darrays=[
                    nib.gifti.GiftiDataArray(mask, datatype='NIFTI_TYPE_INT32')
                ]), gii)
                params[key] = gii

            if verbose:
                print(f'Generating {name}.{hemi} index image: ', params['mni'])
            for cmd in (FSMASK, TEMPTONAT, NATMASK, SURFTOVOL, T1WTOMNI):
                run(cmd.format(**params))
            generated[name].append(params['mni'])
            for key in ('resamp', 'volume', 'out', 'fswhite', 'fspial', 'src',
                        'trg', 'srcmid', 'trgmid', 'natmask', 'fsmask'):
                params[key].unlink()

    return generated


def civet_surf_regfusion(subdir, res='41k', verbose=True):
    """
    Runs registration fusion pipeline on HCP subject `subdir`

    Generates mapping from CIVET --> MNI space

    Parameters
    ----------
    subdir : str or os.PathLike
        Path to HCP subject directory
    res : {'41k'}, optional
        Resolution from which to project the data
    verbose : bool, optional
        Whether to print status messages. Default: True

    Returns
    -------
    generated : dict-of-list
        Dictionary with keys ('x', 'y', 'z') containing lists of left-/right-
        hemisphere projected index maps for registration fusion
    """

    # path handling
    subdir = Path(subdir).resolve()
    subid = subdir.name
    surfdir = subdir / 'surfaces'
    natobj = f'{subid}_{{surf}}_surface_{{hemi}}_81920.obj'
    rslobj = f'{subid}_{{surf}}_surface_rsl_{{hemi}}_81920.obj'
    mnc = subdir / 'final' / f'{subid}_t1_tal.mnc'

    tempdir = Path(tempfile.gettempdir()) / (subid + '_regfusion')
    tempdir.mkdir(exist_ok=True, parents=True)

    # civet medial wall
    medial = fetch_atlas('civet', res)['medial']

    # inverse surface registrations
    surfmap = _civet_surf_reg(subdir, verbose=False)

    # run the actual commands
    generated = defaultdict(list)
    nii = tmpname(suffix='.nii.gz', directory=tempdir)
    nib.save(minc_to_nifti(mnc), nii)
    for hemi, med, surfm in zip(('left', 'right'), medial, surfmap):
        sphere = (subdir / 'gifti'
                  / f'{subid}_sphere_surface_{hemi}_81920.surf.gii')
        for img, name in zip(make_surf_xyz(sphere), ('x', 'y', 'z')):
            template = tmpname(suffix='.shape.gii', directory=tempdir)
            nib.save(img, template)
            params = dict(
                volume=tmpname(prefix=f'{hemi}.', suffix=f'{name}.nii.gz'),
                white=surfdir / natobj.format(surf='white', hemi=hemi),
                pial=surfdir / natobj.format(surf='gray', hemi=hemi),
                trgmid=surfdir / natobj.format(surf='mid', hemi=hemi),
                srcmid=surfdir / rslobj.format(surf='mid', hemi=hemi),
                fsmask=med,
                t1w=nii,
                surfmap=surfm,
                out=tmpname(suffix='.func.gii', directory=tempdir),
                resamp=template
            )

            # we need gifti images!
            for key in ('white', 'pial', 'srcmid', 'trgmid'):
                fn = tempdir / params[key].name
                params[key] = obj_to_gifti(params[key], fn)

            # now generate the reg-fusion outputs
            if verbose:
                print(f'Generating {name}.{hemi} index image: ',
                      params['volume'])

            run(FSMASK.format(**params))

            # resample the data to the native surface
            source = nib.load(params['srcmid']).agg_data()
            morph = nib.load(params['resamp']).agg_data()
            target = nib.load(params['trgmid']).agg_data()
            resamp = resample_surface_map(source, morph, target,
                                          params['surfmap'])
            nib.save(construct_shape_gii(resamp), params['out'])

            # project data to the volume (MNI space!)
            run(SURFTOVOL.format(**params))

            generated[name].append(params['volume'])
            for key in ('white', 'pial', 'srcmid', 'trgmid', 'out', 'resamp'):
                params[key].unlink()

        template.unlink()

    nii.unlink()

    return generated


def _civet_surf_reg(subdir, verbose=True):
    """
    Runs "inverse" CIVET surface registration (mapping model -> native)

    Parameters
    ----------
    subdir : str or os.PathLike
        Path to HCP subject directory

    Returns
    -------
    surfmap : tuple-of-os.PathLike
        Filepaths to generated surface map files
    """

    try:
        import docker
    except ImportError:
        raise ImportError('docker-py package required to generate inverse '
                          'surface registration for CIVET data')

    subdir = Path(subdir).resolve()
    subid = subdir.name

    # don't re-run things if we don't need to!
    expected = [
        subdir / 'transforms' / 'surfreg' / f'{subid}_{hemi}_surfmap_inv.sm'
        for hemi in ('left', 'right')
    ]
    if all(fn.exists() for fn in expected):
        return tuple(expected)

    # get CIVET surface models
    atldir = resource_filename('neuromaps_dev', 'data/civet')

    # get docker client and pull civet image
    client = docker.from_env()
    img = client.images.pull('mcin/civet')

    # run heudiconv over all potential sessions
    generated = tuple()
    for hemi in ('left', 'right'):
        natobj = f'surfaces/{subid}_mid_surface_{hemi}_81920.obj'
        template = f'tpl-civet_den-41k_hemi-{hemi[0].upper()}_midthickness.obj'
        surfmap = f'transforms/surfreg/{subid}_{hemi}_surfmap_inv.sm'
        ses_logs = ''
        cli = client.containers.run(
            image=img,
            command=' '.join([
                '/opt/CIVET/Linux-x86_64/CIVET-2.1.1/progs/bestsurfreg.pl',
                '-clobber', '-min_control_mesh', '5120', '-max_control_mesh',
                '81920', '-blur_coef', '1.5', '-mode', 'stiff',
                '-neighbourhood_radius', '2.8',
                f'/data/{natobj}',
                f'/atlas/{template}',
                f'/data/{surfmap}'
            ]),
            detach=True,
            volumes={str(subdir): {'bind': '/data', 'mode': 'rw'},
                     str(atldir): {'bind': '/atlas', 'mode': 'ro'}}
        )

        # print output to screen but also store it in a logfile
        for log in cli.logs(stream=True):
            curr_log = log.decode()
            ses_logs += curr_log
            if verbose:
                print(curr_log, end='')
        log_file = (subdir / 'logs'
                    / f'{subid}.surface_registration_inv_{hemi}.log')
        with log_file.open(mode='w', encoding='utf-8') as dest:
            dest.write(ses_logs)

        expected = subdir / surfmap

        # if we successfully finished running things then generated the
        # relevant "finished" file
        if expected.exists():
            generated += (expected,)
            finished = (log_file.parent
                        / log_file.name.replace('.log', '.finished'))
            with finished.open(mode='w', encoding='utf-8') as dest:
                pass

    return generated
