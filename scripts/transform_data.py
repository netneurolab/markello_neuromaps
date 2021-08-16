#!/usr/bin/env python

import itertools
import os
from pathlib import Path
import re
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # noqa

from neuromaps import datasets, plotting, resampling, stats, transforms
from neuromaps.datasets.annotations import MATCH

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.sans-serif'] = ['Myriad Pro']
plt.rcParams['font.size'] = 20.0

FIGDIR = Path('./figures/transformed')
OUTDIR = Path('./data/derivatives/correlations')
IMAGES = [
    ('hill2010', 'evoexp', 'fsLR', '164k'),
    ('hill2010', 'devexp', 'fsLR', '164k'),
    ('mueller2013', 'intersubjvar', 'fsLR', '164k'),
    ('raichle', 'cmruglu', 'fsLR', '164k'),
    ('hcps1200', 'myelinmap', 'fsLR', '32k'),
    ('margulies2016', 'fcgradient01', 'fsLR', '32k'),
    ('reardon2018', 'scalingpnc', 'civet', '41k'),
    ('abagen', 'genepc1', 'fsaverage', '10k'),
    ('hcps1200', 'megalpha', 'fsLR', '4k')
]
PARAMS = [
    dict(cmap='caret_blueorange', vmin=-2.7, vmax=2.7),
    dict(cmap='caret_blueorange', vmax=0.5),
    dict(cmap='caret_blueorange', vmin=0.5, threshold=0.5),
    dict(cmap='roy_big_bl', vmin=4500, vmax=7500, threshold=3000),
    dict(cmap='videen_style', vmin=0.45, vmax=1.7, threshold=0.9),
    dict(cmap='jet'),
    dict(cmap='bwr', vmin=0.5, vmax=1.5),
    dict(cmap='rocket', vmax=2),
    dict(cmap='RdBu_r', vmin=0.15)
]
HEMI = re.compile('hemi-(L|R)')


def correlate_images(images, resamp):
    """
    Resamples and correlates all pairs of datasets in `images`

    Parameters
    ----------
    images : (N,) list-of-tuple of str or os.PathLike
        Datasets to be resampled + correlation
    resamp : str
        Method to resample pairs of datasets

    Returns
    -------
    corrs : (N * (N - 1) / 2,) np.ndarray
        Correlations between pairs of `images`
    """

    images = images.copy()
    for i, img in enumerate(images):
        source, desc, space, den = img
        images[i] = datasets.fetch_annotation(source=source, desc=desc,
                                              space=space, den=den)[img]

    n, n_images = 0, len(images)
    corrs = np.zeros(int((n_images * (n_images - 1) / 2)))
    for im1, im2 in itertools.combinations(images, 2):
        src_space = MATCH.search(str(im1[0])).group(3)
        trg_space = MATCH.search(str(im2[0])).group(3)

        # handle single-hemisphere data (rather inelegantly...)
        hemi = None
        if len(im1) == 1 and len(im2) != 1:
            hemi = HEMI.search(str(im1[0])).group(1)
            im2 = tuple(
                i for i in im2 if re.search(f'hemi-{hemi}', i) is not None
            )
        elif len(im1) != 1 and len(im2) == 1:
            hemi = HEMI.search(str(im2[0])).group(1)
            im1 = tuple(
                i for i in im1 if re.search(f'hemi-{hemi}', i) is not None
            )
        elif len(im1) == 1 and len(im2) == 1:
            if HEMI.search(im1[0]).group(1) != HEMI.search(im2[0]).group(1):
                corrs[n] = np.nan
                continue

        # resample and correlate images
        im1, im2 = resampling.resample_images(im1, im2, src_space, trg_space,
                                              hemi=hemi, resampling=resamp)
        corrs[n] = stats.correlate_images(im1, im2)
        n += 1

    return corrs


def resample_and_plot(image, trg_space='fsLR', trg_density='32k', **kwargs):
    """
    Resamples `image` to `trg_space` / `trg_density` and plots / saves figure

    Parameters
    ----------
    image : str or os.PathLike
        Filepath to single hemisphere image
    trg_space : str, optional
        Target space to resample `image` to. Default: 'fsLR'
    trg_density : str, optional
        Target density to resample `image` to. Default: '32k'
    kwargs : key-value pairs
        Plotting parameters passed directly to `plotting.plot_surf_template`

    Returns
    -------
    fn : str
        Filepath to save plot figure
    """

    if (isinstance(image, (str, os.PathLike))
            or not isinstance(image, Iterable)):
        image = (image,)

    # if only one annotation provided get the relevant hemisphere
    hemi = None
    if len(image) == 1:
        hemi = re.search('hemi-(L|R)', str(image[0])).group(1)

    # no need to resample if we're already trg_space / trg_density
    resampled = image
    space, density = MATCH.search(str(image[0])).groups()[2:]
    if space != trg_space or density != trg_density:
        func = getattr(transforms, f'{space.lower()}_to_{trg_space.lower()}')
        resampled = func(image, density, hemi=hemi)

    fig = plotting.plot_surf_template(resampled, trg_space, trg_density,
                                      hemi=hemi, surf='veryinflated', **kwargs)
    fn = FIGDIR / f'{str(image[0].name)[:-9]}.png'
    fig.savefig(fn, dpi=300, transparent=True, bbox_inches='tight')
    plt.close(fig=fig)

    return fn


def main():
    for fd in (FIGDIR, OUTDIR):
        fd.mkdir(exist_ok=True, parents=True)

    for n, image in enumerate(IMAGES):
        source, desc, space, den = image
        image = datasets.fetch_annotation(source=source, desc=desc,
                                          space=space, den=den)[image]
        resample_and_plot(image, **PARAMS[n])

    allcorrs = []
    funcs = (
        'transform_to_src', 'transform_to_trg', 'transform_to_alt',
        'downsample_only'
    )
    for func in funcs:
        corrs = correlate_images(IMAGES, func)
        np.savetxt(OUTDIR / f'corrs_{func}.txt', corrs)
        allcorrs.append(corrs)

    allcorrs = np.column_stack(allcorrs)
    labels = [f.replace('_', ' ') for f in funcs]
    ax = sns.heatmap(allcorrs[allcorrs.mean(1).argsort()[::-1]],
                     xticklabels=labels, yticklabels=[],
                     vmin=-0.5, vmax=0.75, center=0, cmap='RdBu_r',
                     cbar_kws={'label': 'correlation ($r$)'})
    ax.tick_params(width=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylabel(r'$r_{\mathrm{map1}, \mathrm{map2}}$')
    for suff in ('png', 'svg'):
        ax.figure.savefig(FIGDIR / f'corrs.{suff}',
                          dpi=300, transparent=True, bbox_inches='tight')


if __name__ == "__main__":
    main()
