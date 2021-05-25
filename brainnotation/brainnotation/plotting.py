# -*- coding: utf-8 -*-
"""
Functionality for plotting
"""

from pathlib import Path
from pkg_resources import resource_filename

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from nilearn import plotting

from brainnotation.transforms import DENSITIES

CIVETDIR = Path('/home/rmarkello/data/civet')
ATLASDIR = Path(resource_filename('brainnotation', 'data/atlases'))
REFMESH = resource_filename('brainnotation', 'data/fsaverage.{hemi}_LR.'
                            'spherical_std.164k_fs_LR.surf.gii')
REFSULC = resource_filename('brainnotation', 'data/{hemi}.refsulc.164k_fs_LR.'
                            'shape.gii')


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
        plotting.plot_surf(tmpdir / fmt.format(hemi=hm, desc=''),
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
            ax = plotting.plot_surf(
                str(sphere),
                str(CIVETDIR / sub / 'gifti' / sulc),
                hemi=hemi, axes=ax
            )
    fig.tight_layout()

    return fig


def plot_to_template(data, space, density, surf='inflated', **kwargs):
    """
    Plots `data` on template `space`

    Parameters
    ----------
    data : str or os.PathLike or tuple-of-str
        Path to data file(s) to be plotted. If tuple, assumes (left, right)
        hemisphere.
    space : {'civet', 'fsaverage', 'fsLR'}
        Space in which `data` is defined
    density : str
        Resolution of template
    surf : str, optional
        Surface on which `data` should be plotted. Must be valid for specified
        `space`. Default: 'inflated'
    kwargs : key-value pairs
        Passed directly to `nilearn.plotting.plot_surf`

    Returns
    -------
    fig : matplotlib.Figure instance
        Plotted figure
    """

    if space not in DENSITIES or space == 'MNI152':
        raise ValueError('Invalid space argument')
    if density not in DENSITIES[space]:
        raise ValueError('Invalid density argument')

    fmt = ATLASDIR / space \
        / f'tpl-{space}_den-{density}_hemi-{{hemi}}_{surf}.surf.gii'

    n_surf = 1 if len(data) != 2 else 2
    fig, axes = plt.subplots(1, n_surf, subplot_kw={'projection': '3d'})
    if n_surf == 1:
        data, axes = (data,), (axes,)
    for ax, hemi, img in zip(axes, ('left', 'right'), data):
        template = str(fmt.parent / fmt.name.format(hemi=hemi[0].upper()))
        plotting.plot_surf(template, str(img), hemi=hemi, axes=ax, **kwargs)
    fig.tight_layout()

    return fig
