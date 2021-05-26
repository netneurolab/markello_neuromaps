# -*- coding: utf-8 -*-
"""
Functionality for plotting
"""

from pathlib import Path
from brainnotation.images import load_gifti
from pkg_resources import resource_filename

from matplotlib.cm import register_cmap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from nilearn.plotting import plot_surf
import numpy as np
import seaborn as sns

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
register_cmap('caret_blueorange', sns.blend_palette([
        '#00d2ff', '#009eff', '#006cfe', '#0043fe',
        '#fd4604', '#fe6b01', '#ffd100', '#ffff04'
    ], as_cmap=True)
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
                     hemi=None, shading=0.6, **kwargs):
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
    medial = ATLASDIR / template \
        / f'tpl-{template}_den-{density}_hemi-{{hemi}}_desc-nomedialwall_' \
        'dparc.label.gii'

    opts = dict(threshold=np.spacing(1), alpha=1.0)
    opts.update(**kwargs)
    if opts.get('bg_map') is not None and kwargs.get('alpha') is None:
        opts['alpha'] = 'auto'

    data, hemi = zip(*_check_hemi(data, hemi))
    n_surf = len(data)
    fig, axes = plt.subplots(n_surf, 2, subplot_kw={'projection': '3d'})
    if n_surf == 1:
        axes = (axes,)
    for row, h, img in zip(axes, hemi, data):
        img = load_gifti(img).agg_data()
        med = load_gifti(str(medial).format(hemi=h)).agg_data()
        geom = load_gifti(str(fmt).format(hemi=h)).agg_data()

        for ax, view in zip(row, ['lateral', 'medial']):
            plot_surf(geom, img, hemi=HEMI[h], axes=ax, view=view, **opts)
            ax.collections[0].set_facecolors(
                _fix_facecolors(
                    ax.collections[0]._original_facecolor, *geom, med, shading
                )
            )

    if not opts.get('colorbar', False):
        fig.tight_layout()

    return fig


def _fix_facecolors(facecolors, vertices, faces, medial, shading=0.6):
    """
    Updates `facecolors` to reflect shading of mesh geometry + medial wall

    Parameters
    ----------
    facecolors : (F,) array_like
        Original facecolors of plot
    vertices : (V, 3)
        Vertices of surface mesh
    faces : (F, 3)
        Triangles of surface mesh
    medial : (V,) array_like
        Boolean array wherere 0 indicates the medial wall
    shading : (0, 1), optional
        Shading intensity (0 = no shading, 1 = dark). Default: 0.7

    Returns
    -------
    fc : (F,) array_like
        Updated facecolors with approriate shading + medial wall removed
    """

    fc = np.asarray(facecolors).copy()
    fc[np.any(np.logical_not(medial)[faces], axis=1)] = plt.cm.gray_r(0.5)
    fc[:, :-1] *= _shading_intensity(vertices, faces, shading=shading)[:, None]

    return fc


def _shading_intensity(vertices, faces, shading=0.7):
    """
    Generates intensity values to modify shading of triangle faces

    Parameters
    ----------
    vertices : (V, 3)
        Vertices of surface mesh
    faces : (F, 3)
        Triangles of surface mesh
    shading : (0, 1), optional
        Shading intensity (0 = no shading, 1 = dark). Default: 0.7

    Returns
    -------
    intensity :
    """

    tris = vertices[faces]
    norms = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norms /= np.linalg.norm(norms, axis=1, keepdims=True)
    intensity = norms @ np.array([0, 0, 1])
    intensity[np.isnan(intensity)] = 1
    min_int = np.min(intensity)
    intensity = ((1 - shading)
                 + (shading * (intensity - min_int)
                    / ((np.percentile(intensity, 80) - min_int))))

    return np.clip(intensity, None, 1)
