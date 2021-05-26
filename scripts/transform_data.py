#!/usr/bin/env python

import itertools
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # noqa

from brainnotation import plotting, transforms
from brainnotation.images import load_gifti

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.sans-serif'] = ['Myriad Pro']
plt.rcParams['font.size'] = 20.0

ATLASDIR = Path('./data/raw/brain_maps')
MATCH = re.compile(r'_space-(\S+)_den-(\d+k)_hemi-(\S+)_')
IMAGES = [
   ATLASDIR / 'fsLR' / '164k' / 'source-hill2010_desc-evoexp_space-fsLR_den-164k_hemi-R_feature.func.gii',  # noqa
   ATLASDIR / 'fsLR' / '164k' / 'source-hill2010_desc-devexp_space-fsLR_den-164k_hemi-R_feature.func.gii',  # noqa
   ATLASDIR / 'fsLR' / '164k' / 'source-mueller2013_desc-intersubjvar_space-fsLR_den-164k_hemi-L_feature.func.gii',  # noqa
   ATLASDIR / 'fsLR' / '164k' / 'source-raichle_desc-cmruglu_space-fsLR_den-164k_hemi-L_feature.func.gii',  # noqa
   ATLASDIR / 'fsLR' / '32k' / 'source-hcps1200_desc-myelinmap_space-fsLR_den-32k_hemi-L_feature.func.gii',  # noqa
   ATLASDIR / 'fsLR' / '32k' / 'source-margulies2016_desc-fcgradient01_space-fsLR_den-32k_hemi-L_feature.func.gii',  # noqa
   ATLASDIR / 'civet' / '41k' / 'source-reardon2018_desc-scalingpnc_space-civet_den-41k_hemi-L_feature.func.gii',  # noqa
   ATLASDIR / 'fsaverage' / '10k' / 'source-abagen_desc-genepc1_space-fsaverage_den-10k_hemi-L_feature.func.gii',  # noqa
   ATLASDIR / 'fsLR' / '4k' / 'source-hcps1200_desc-megalpha_space-fsLR_den-4k_hemi-L_feature.func.gii'  # noqa
]
MAPPING = dict(
    fsLR=transforms.fslr_to_fsaverage,
    civet=transforms.civet_to_fsaverage,
    fsaverage=transforms.fsaverage_to_fsaverage
)


def imgcorr(src, trg):
    srcdata = np.nan_to_num(load_gifti(src).agg_data())
    trgdata = np.nan_to_num(load_gifti(trg).agg_data())
    mask = np.logical_and(np.isclose(srcdata, 0), np.isclose(trgdata, 0))
    return np.corrcoef(srcdata[~mask], trgdata[~mask])[0, 1]


def downsample_only(src, trg):
    srcspace, srcden, srchemi = MATCH.search(str(src)).groups()
    trgspace, trgden, trghemi = MATCH.search(str(trg)).groups()
    srcnum, trgnum = int(srcden[:-1]), int(trgden[:-1])
    # resample to `trg`
    if int(srcden[:-1]) >= int(trgden[:-1]):
        # if images are not same resolution or same space, transform
        # otherwise, they're identical res + space so we're good to go
        if srcspace != trgspace or srcnum != trgnum:
            func = getattr(transforms,
                           f'{srcspace.lower()}_to_{trgspace.lower()}')
            src, = func(src, srcden, trgden, hemi=trghemi)
    # resample to `src`
    elif int(srcden[:-1]) < int(trgden[:-1]):
        func = getattr(transforms, f'{trgspace.lower()}_to_{srcspace.lower()}')
        trg, = func(trg, trgden, srcden, hemi=trghemi)

    return imgcorr(src, trg)


def transform_to_src(src, trg):
    srcspace, srcden, srchemi = MATCH.search(str(src)).groups()
    trgspace, trgden, trghemi = MATCH.search(str(trg)).groups()
    func = getattr(transforms, f'{trgspace.lower()}_to_{srcspace.lower()}')
    trg, = func(trg, trgden, srcden, hemi=trghemi)

    return imgcorr(src, trg)


def transform_to_trg(src, trg):
    srcspace, srcden, srchemi = MATCH.search(str(src)).groups()
    trgspace, trgden, trghemi = MATCH.search(str(trg)).groups()
    func = getattr(transforms, f'{srcspace.lower()}_to_{trgspace.lower()}')
    src, = func(src, srcden, trgden, hemi=srchemi)

    return imgcorr(src, trg)


def transform_to_alt(src, trg, space='fsaverage', den='41k'):
    srcspace, srcden, srchemi = MATCH.search(str(src)).groups()
    trgspace, trgden, trghemi = MATCH.search(str(trg)).groups()
    func = getattr(transforms, f'{srcspace.lower()}_to_{space.lower()}')
    src, = func(src, srcden, den, hemi=srchemi)
    func = getattr(transforms, f'{trgspace.lower()}_to_{space.lower()}')
    trg, = func(trg, trgden, den, hemi=trghemi)

    return imgcorr(src, trg)


def correlate_images(images, func):
    n, n_images = 0, len(images)
    corrs = np.zeros(int((n_images * (n_images - 1) / 2)))
    for im1, im2 in itertools.combinations(images, 2):
        corrs[n] = func(im1, im2)
        n += 1

    return corrs


def implot(image):
    space, density, hemi = MATCH.search(str(image)).groups()
    out = MAPPING[space](image, density, hemi=hemi)
    fig = plotting.plot_to_template(out, 'fsaverage', '41k', hemi=hemi,
                                    cmap='rocket')
    fig.savefig(f'/home/rmarkello/Desktop/{str(image.name)[:-9]}.png',
                dpi=300, transparent=True, bbox_inches='tight')
    plt.close(fig=fig)


def main():
    for image in IMAGES:
        implot(image)

    allcorrs = []
    funcs = [
        transform_to_src, transform_to_trg, transform_to_alt, downsample_only
    ]
    for func in funcs:
        print(func.__name__)
        corrs = correlate_images(IMAGES[2:], func)
        np.savetxt(f'/home/rmarkello/Desktop/corrs_{func.__name__}.txt', corrs)
        allcorrs.append(corrs)

    allcorrs = np.column_stack(allcorrs)
    labels = [f.__name__.replace('_', ' ') for f in funcs]
    ax = sns.heatmap(allcorrs[allcorrs.mean(1).argsort()[::-1]],
                     xticklabels=labels, yticklabels=[],
                     vmin=-0.5, vmax=0.75, center=0, cmap='RdBu_r',
                     cbar_kws={'label': 'correlation ($r$)'})
    ax.tick_params(width=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylabel(r'$r_{\mathrm{map1}, \mathrm{map2}}$')
    for suff in ('png', 'svg'):
        ax.figure.savefig(f'/home/rmarkello/Desktop/corrs.{suff}',
                          dpi=300, transparent=True, bbox_inches='tight')


if __name__ == "__main__":
    main()
