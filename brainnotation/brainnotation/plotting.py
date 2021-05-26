# -*- coding: utf-8 -*-
"""
Functionality for plotting
"""

from pathlib import Path
from pkg_resources import resource_filename

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import nibabel as nib
from nilearn.plotting import plot_surf
import numpy as np

from brainnotation.transforms import DENSITIES, _check_hemi

CIVETDIR = Path('/home/rmarkello/data/civet')
ATLASDIR = Path(resource_filename('brainnotation', 'data/atlases'))
REFMESH = resource_filename('brainnotation', 'data/fsaverage.{hemi}_LR.'
                            'spherical_std.164k_fs_LR.surf.gii')
REFSULC = resource_filename('brainnotation', 'data/{hemi}.refsulc.164k_fs_LR.'
                            'shape.gii')
HEMI = dict(L='left', R='right')
ALIAS = dict(
    fslr='fsLR', fsavg='fsaverage', mni152='MNI152', mni='MNI152'
)


def plot_fslr_sulc():
    """
    Plots fs_LR reference sulcal depth maps

    Returns
    -------
    fig : matplotlib.Figure
        Plotted figure
    """

    fig, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'})

    tmpdir = ATLASDIR / 'fsLR'
    fmt = 'tpl-fsLR_den-164k_hemi-{{hemi}}{{desc}}_sphere.surf.gii'

    for ax, hemi in zip(axes, ('left', 'right')):
        hm = hemi[0].upper()
        plot_surf(tmpdir / fmt.format(hemi=hm, desc=''),
                  tmpdir / fmt.format(hemi=hm, desc='_desc-sulc'),
                  hemi=hemi, axes=ax)
    fig.tight_layout()

    return fig


def plot_civet_msm(subjdir, rot=True):
    """
    Plots original + rotated CIVET spheres w/sulcal depth information

    Parameters
    ----------
    subjdir : str or os.PathLike
        Path to data directory where `msmrot` and `msmsulc` directories live
    rot : bool, optional
        Whether to plot outputs from `msmrot` (True) or `msmsulc` (False).
        Default: True

    Returns
    -------
    fig : matplotlib.Figure
        Plotted figure
    """

    subjdir = Path(subjdir)
    sub = subjdir.name
    fig, axes = plt.subplots(2, 2, subplot_kw={'projection': '3d'})
    for n, row in enumerate(axes):
        for ax, hemi in zip(row, ('left', 'right')):
            if n:
                msmdir = 'msmrot' if rot else 'msmsulc'
                sphere = (subjdir / msmdir
                          / f'{hemi[0].upper()}.sphere.reg.surf.gii')
            else:
                sphere = (CIVETDIR / sub / 'gifti'
                          / f'{sub}_sphere_surface_rsl_{hemi}_81920.surf.gii')
            sulc = f'{sub}_sulc_surface_rsl_{hemi}_81920.shape.gii'
            ax = plot_surf(str(sphere), str(CIVETDIR / sub / 'gifti' / sulc),
                           hemi=hemi, axes=ax)
    fig.tight_layout()

    return fig


def plot_to_template(data, template, density, surf='inflated', space=None,
                     hemi=None, **kwargs):
    """
    Plots `data` on `template` surface

    Parameters
    ----------
    data : str or os.PathLike or tuple-of-str
        Path to data file(s) to be plotted. If tuple, assumes (left, right)
        hemisphere.
    template : {'civet', 'fsaverage', 'fsLR'}
        Template on which `data` is defined
    density : str
        Resolution of template
    surf : str, optional
        Surface on which `data` should be plotted. Must be valid for specified
        `space`. Default: 'inflated'
    space : str, optional
        Space `template` is aligned to. Default: None
    kwargs : key-value pairs
        Passed directly to `nilearn.plotting.plot_surf`

    Returns
    -------
    fig : matplotlib.Figure instance
        Plotted figure
    """

    template = ALIAS.get(template, template)
    if template not in DENSITIES or template == 'MNI152':
        raise ValueError('Invalid space argument')
    if density not in DENSITIES[template]:
        raise ValueError('Invalid density argument')
    space = f'_space-{space}' if space is not None else ''

    fmt = ATLASDIR / template \
        / f'tpl-{template}{space}_den-{density}_hemi-{{hemi}}_{surf}.surf.gii'
    bg_map = ATLASDIR / template \
        / f'tpl-{template}_den-{density}_hemi-{{hemi}}_desc-sulc_' \
        'midthickness.shape.gii'

    data, hemi = zip(*_check_hemi(data, hemi))
    n_surf = len(data)
    fig, axes = plt.subplots(n_surf, 2, subplot_kw={'projection': '3d'})
    if n_surf == 1:
        axes = (axes,)
    for row, h, img in zip(axes, hemi, data):
        if isinstance(img, nib.gifti.GiftiImage):
            img = img.agg_data()
        template = str(fmt.parent / fmt.name.format(hemi=h))
        sulc = bg_map.parent / bg_map.name.format(hemi=h)
        opts = dict(bg_map=str(sulc) if sulc.exists() else None,
                    threshold=np.spacing(1))
        opts.update(**kwargs)
        for ax, view in zip(row, ['lateral', 'medial']):
            plot_surf(template, img, hemi=HEMI[h], axes=ax, view=view, **opts)
    fig.tight_layout()

    return fig
