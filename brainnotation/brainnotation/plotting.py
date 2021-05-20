from pathlib import Path
from pkg_resources import resource_filename

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from nilearn import plotting

CIVETDIR = Path('/home/rmarkello/data/civet')
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
    for ax, hemi in zip(axes, ('left', 'right')):
        plotting.plot_surf(REFMESH.format(hemi=hemi[0].upper()),
                           REFSULC.format(hemi=hemi[0].upper()),
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
