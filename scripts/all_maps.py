#!/usr/bin/env python

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from nilearn.plotting import plot_glass_brain
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
import seaborn as sns  # noqa

from neuromaps.datasets import fetch_atlas, fetch_annotation
from neuromaps import images, transforms
import neuromaps.plotting # noqa
from surfplot import Plot

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.sans-serif'] = ['Myriad Pro']
plt.rcParams['font.size'] = 20.0

DATAPATH = Path('./data/derivatives').resolve()
OUTPATH = Path('/home/rmarkello/Desktop/neuromap_figures')
OUTPATH.mkdir(exist_ok=True, parents=True)
ANNOTATIONS = [
    ('abagen', 'genepc1', 'fsaverage', '10k'),
    ('hcps1200', 'megalpha', 'fsLR', '4k'),
    ('hcps1200', 'megbeta', 'fsLR', '4k'),
    ('hcps1200', 'megdelta', 'fsLR', '4k'),
    ('hcps1200', 'meggamma1', 'fsLR', '4k'),
    ('hcps1200', 'meggamma2', 'fsLR', '4k'),
    ('hcps1200', 'megtheta', 'fsLR', '4k'),
    ('hcps1200', 'myelinmap', 'fsLR', '32k'),
    ('hcps1200', 'thickness', 'fsLR', '32k'),
    ('hill2010', 'devexp', 'fsLR', '164k'),
    ('hill2010', 'evoexp', 'fsLR', '164k'),
    ('margulies2016', 'fcgradient01', 'fsLR', '32k'),
    ('mueller2013', 'intersubjvar', 'fsLR', '164k'),
    ('neurosynth', 'cogpc1', 'MNI152', '2mm'),
    ('raichle', 'cbf', 'fsLR', '164k'),
    ('raichle', 'cbv', 'fsLR', '164k'),
    ('raichle', 'cmr02', 'fsLR', '164k'),
    ('raichle', 'cmruglu', 'fsLR', '164k'),
    ('reardon2018', 'scalingnih', 'civet', '41k'),
    ('reardon2018', 'scalingpnc', 'civet', '41k'),
]
PARAMS = [
    dict(cmap='rocket', color_range=(-1.75, 2)),
    dict(cmap='RdBu_r'),
    dict(cmap='RdBu_r'),
    dict(cmap='RdBu_r'),
    dict(cmap='RdBu_r'),
    dict(cmap='RdBu_r'),
    dict(cmap='RdBu_r'),
    dict(cmap='videen_style', color_range=(0.45, 1.7)),
    dict(cmap='videen_style', color_range=(0, 3.3)),
    dict(cmap='caret_blueorange', color_range=(-0.4, 0.5)),
    dict(cmap='caret_blueorange', color_range=(-2.7, 2.7)),
    dict(cmap='jet'),
    dict(cmap='caret_blueorange', color_range=(0.5, 0.70)),
    dict(cmap='videen_style', color_range=(2000, 7000)),
    dict(cmap='videen_style', color_range=(0, 6000)),
    dict(cmap='videen_style', color_range=(2000, 7000)),
    dict(cmap='roy_big_bl', color_range=(4500, 7500)),
    dict(cmap='bwr', color_range=(0.5, 1.5)),
    dict(cmap='bwr', color_range=(0.5, 1.5)),
]


def savefig(fig, fn):
    fig.savefig(OUTPATH / fn, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig=fig)


def main():
    for (src, desc, atlas, res), (params) in zip(ANNOTATIONS, PARAMS):
        annot = fetch_annotation(source=src, desc=desc, return_single=True)

        if atlas == 'MNI152':
            annot = transforms.mni152_to_fsaverage(annot, '10k')
            atlas = 'fsaverage'
            res = '10k'

        atlas = fetch_atlas(atlas, res)

        try:
            lh, rh = atlas['veryinflated']
        except KeyError:
            lh, rh = atlas['inflated']

        if not isinstance(annot, list) or len(annot) != 2:
            p = Plot(surf_rh=rh, zoom=1.2)
            p.add_layer(annot, **params)
        else:
            p = Plot(surf_lh=lh, surf_rh=rh)
            p.add_layer({'left': annot[0], 'right': annot[1]}, **params)

        savefig(p.build(), f'{src}_{desc}.png')


def cogpc1():
    annot = fetch_annotation(desc='cogpc1', return_single=True)
    fig = plot_glass_brain(annot, cmap='rocket', display_mode='x')
    savefig(fig.axes['x'].ax.figure, 'cogpc1_mni152_2mm.png')

    atlases = ('civet', 'fsaverage', 'fslr')
    densities = (['41k'], ['10k', '41k', '164k'], ['32k', '164k'])
    for atlas, dens in zip(atlases, densities):
        for den in dens:
            surf = fetch_atlas(atlas, den)
            try:
                lh, rh = surf['veryinflated']
            except KeyError:
                lh, rh = surf['inflated']
            p = Plot(surf_lh=lh, surf_rh=rh)
            func = getattr(transforms, f'mni152_to_{atlas}')
            out = func(annot, den)
            p.add_layer({'left': out[0], 'right': out[1]}, cmap='rocket')
            savefig(p.build(), f'cogpc1_{atlas}_{den}.png')


def genepc1():
    annot = fetch_annotation(desc='genepc1', return_single=True)

    atlases = ('civet', 'fsaverage', 'fslr')
    densities = (['41k'], ['10k', '41k', '164k'], ['32k', '164k'])
    for atlas, dens in zip(atlases, densities):
        for den in dens:
            surf = fetch_atlas(atlas, den)
            try:
                lh, rh = surf['veryinflated']
            except KeyError:
                lh, rh = surf['inflated']
            p = Plot(surf_lh=lh, surf_rh=rh)
            func = getattr(transforms, f'fsaverage_to_{atlas}')
            out = func(annot, '10k', den)
            p.add_layer({'left': out[0], 'right': out[1]}, cmap='rocket',
                        color_range=(-1.75, 2))
            savefig(p.build(), f'genepc1_{atlas}_{den}.png')


def get_map_correlations(atlas='fsLR', density='32k'):
    path = DATAPATH / 'correlations' / atlas / density / 'correlations.csv'
    data = pd.read_csv(path)
    labels = ['-'.join(annot) for annot in ANNOTATIONS]
    mask = (data['annot1'].apply(lambda x: x in labels)
            & data['annot2'].apply(lambda x: x in labels))

    return data.loc[mask]


def draw_network(atlas='fsLR', density='32k'):
    data = get_map_correlations(atlas, density)
    corrs = squareform(data['corr'])
    corrs[corrs < 0.3] = 0

    network = nx.from_numpy_array(corrs)
    pos = nx.spring_layout(network, seed=1234, k=0.2)
    colors = list(nx.get_edge_attributes(network, 'weight').values())
    labels = ['-'.join(annot) for annot in ANNOTATIONS]
    nx.draw_networxk(network, pos, labels=labels, node_size=1000,
                     edge_color=colors, edge_cmap=plt.cm.gray_r)

    fig = plt.gcf()
    savefig(fig, 'brainmap_network.svg')


def distplot_correlations():
    corrs = pd.DataFrame()
    atlases = ('civet', 'fsaverage', 'fsLR')
    densities = (['41k'], ['10k', '41k', '164k'], ['32k', '164k'])
    for atlas, dens in zip(atlases, densities):
        for den in dens:
            data = get_map_correlations(atlas, den)
            corrs = corrs.append(
                pd.DataFrame({'corrs': data['corr'], 'pval': data['pval'],
                              'atlas': atlas, 'den': den})
            )
    corrs = corrs.reset_index(drop=True)


def diff_scatterplot():
    annot1 = fetch_annotation(desc='fcgradient01', return_single=True)
    annot1_civet = transforms.fslr_to_civet(annot1, '32k', '41k')

    annot2 = fetch_annotation(desc='scalingnih', return_single=True)
    annot2_fslr = transforms.civet_to_fslr(annot2, '41k', '32k')

    # idx = 242
    # fslr32k = get_map_correlations('fsLR', '32k').loc[idx]
    # civet41k = get_map_correlations('civet', '41k').loc[idx]

    for im1, im2, atlas in ((annot1, annot2_fslr, 'fslr'),
                            (annot1_civet, annot2, 'civet')):
        im1, im2 = images.load_data(im1), images.load_data(im2)
        mask = np.logical_not(np.isclose(im1, 0) | np.isclose(im2, 0))
        fig, ax = plt.subplots(1, 1)
        ax.hexbin(x=im1[mask], y=im2[mask], mincnt=5, linewidth=0.2,
                  bins='log', alpha=1.0, rasterized=True)
        sns.despine(ax=ax)
        savefig(fig, f'scatter_{atlas}.svg')
