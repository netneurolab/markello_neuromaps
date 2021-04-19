# -*- coding: utf-8 -*-
"""
Contains code for generating registration-fusion mappings
"""

from collections import defaultdict
import os
from pathlib import Path
from pkg_resources import resource_filename
import tempfile

import boto3
from botocore.exceptions import ClientError
import nibabel as nib
import numpy as np

from netneurotools.freesurfer import check_fs_subjid
from netneurotools.utils import run

VOLTOSURF = 'wb_command -volume-to-surface-mapping {volume} {srcmid} ' \
            '{out} -ribbon-constrained {white} {pial} -interpolate TRILINEAR'
SURFTOSURF = 'wb_command -metric-resample {out} {src} {trg} ADAP_BARY_AREA ' \
             '{resamp} -area-surfs {srcmid} {trgmid} -current-roi {natmask}'
NATMASK = 'wb_command -metric-mask {out} {natmask} {out}'
FSMASK = 'wb_command -metric-mask {resamp} {fsmask} {resamp}'
VOLRESAMP = 'wb_command -volume-resample {mni} {space} CUBIC {volume} ' \
            '-warp {warp} -fnirt {space}'
GENMIDTHICK = 'wb_command -surface-average {mid} -surf {white} -surf {pial}'
FSTOGII = 'mris_convert {fs} {gii}'


def tmpname(suffix, prefix=None):
    """
    Little helper function because :man_shrugging:

    Parameters
    ----------
    suffix : str
        Suffix of created filename

    Returns
    -------
    fn : str
        Temporary filename; user is responsible for deletion
    """

    fd, fn = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    os.close(fd)

    return Path(fn)


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
        for data in np.split(xyz.astype('int32'), 3, axis=-1)
    ]
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

    fn = resource_filename('brainnotation', 'data/hcpfiles.txt')
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
        for hemi in ('L', 'R'):
            prefix = f'{subid}.{hemi}'
            params = dict(
                volume=template,
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
            for cmd in (VOLTOSURF, NATMASK, SURFTOSURF, FSMASK):
                run(cmd.format(**params))
            generated[name].append(params['resamp'])
            params['out'].unlink()
        template.unlink()

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
    resolutions = ('fsaverage', 'fsaverage4', 'fsaverage5', 'fsaverage6')
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
        run(VOLRESAMP.format(mni=template,
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
                mask = np.zeros(n_vert)
                mask[nib.freesurfer.read_label(params[key])] = 1
                gii = tmpname(prefix=f'{hemi}.',
                              suffix=f'.{name}.{key}.shape.gii')
                nib.save(nib.GiftiImage(darrays=[
                    nib.gifti.GiftiDataArray(mask,
                                             intent='NIFTI_INTENT_NORMAL',
                                             datatype='NIFTI_TYPE_FLOAT32')
                ]), gii)
                params[key] = gii

            # now generate the reg-fusion mapping (subject -> fsaverage)
            if verbose:
                print(f'Generating {name}.{hemi} index image: '
                      f'{params["resamp"]}')
            for cmd in (VOLTOSURF, NATMASK, SURFTOSURF, FSMASK):
                run(cmd.format(**params))
            generated[name].append(params['resamp'])
            for key in ('out', 'fswhite', 'fspial', 'src', 'trg',
                        'srcmid', 'trgmid', 'natmask', 'fsmask'):
                params[key].unlink()
        volume.unlink()

    return generated


def civet_regfusion(subdir):
    """
    Runs registration fusion pipeline on HCP subject `subdir`

    Generates mapping from MNI --> CIVET space

    Parameters
    ----------
    subdir : str or os.PathLike
        Path to HCP subject directory

    Returns
    -------
    generated : dict-of-list
        Dictionary with keys ('x', 'y', 'z') containing lists of left-/right-
        hemisphere projected index maps for registration fusion
    """

    raise NotImplementedError


def group_regfusion(images):
    """
    Generates group-level registration fusion mappings

    Parameters
    ----------
    images : list of dict
        Outputs from one of the _regfusion() functions for multiple subjects

    Returns
    -------
    mappings : np.ndarray
        Registration fusion mappings from `images`
    """

    raise NotImplementedError
