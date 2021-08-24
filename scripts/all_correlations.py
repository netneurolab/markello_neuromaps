#!/usr/bin/env python

import itertools
from pathlib import Path
import re

import pandas as pd
import numpy as np

from neuromaps import datasets, images, nulls, resampling, stats
from neuromaps.datasets.annotations import MATCH

FIGDIR = Path('./figures/transformed')
OUTDIR = Path('./data/derivatives/correlations')
HEMI = re.compile('hemi-(L|R)')
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
    ('reardon2018', 'scalinghcp', 'civet', '41k'),
    ('reardon2018', 'scalingnih', 'civet', '41k'),
    ('reardon2018', 'scalingpnc', 'civet', '41k'),
    ('satterthwaite2014', 'meancbf', 'MNI152', '1mm'),
]


def correlate_images(annotations, alt_spec, nullmaps):
    """
    Resamples and correlates all pairs of datasets in `annotations`

    Parameters
    ----------
    annotations : (N,) list-of-tuple of str or os.PathLike
        Datasets to be resampled + correlation

    Returns
    -------
    corrs : (N * (N - 1) / 2,) np.ndarray
        Correlations between pairs of `annotations`
    """

    df = []
    for im1, im2 in itertools.combinations(annotations, 2):
        annot1 = MATCH.search(str(im1)).groups()
        annot2 = MATCH.search(str(im2)).groups()
        src_space, trg_space = annot1[2], annot2[2]
        annot1, annot2 = '-'.join(annot1), '-'.join(annot2)
        print(*alt_spec, annot1, annot2)

        # handle single-hemisphere data (rather inelegantly...)
        hemi = None
        if (isinstance(im1, str) and not isinstance(im2, str)
                and src_space != 'MNI152'):
            hemi = HEMI.search(im1).group(1)
            im2 = tuple(
                i for i in im2 if re.search(f'hemi-{hemi}', i) is not None
            )
            im1 = [im1]
        elif (not isinstance(im1, str) and isinstance(im2, str)
                and trg_space != 'MNI152'):
            hemi = HEMI.search(im2).group(1)
            im1 = tuple(
                i for i in im1 if re.search(f'hemi-{hemi}', i) is not None
            )
            im2 = [im2]
        elif (isinstance(im1, str) and isinstance(im2, str)
                and src_space != 'MNI152' and trg_space != 'MNI152'):
            if HEMI.search(im1).group(1) != HEMI.search(im2).group(1):
                corr = pval = np.nan
                df.append([annot1, annot2, corr, pval])
                continue
            hemi = HEMI.search(im1).group(1)
            im1, im2 = [im1], [im2]
        elif isinstance(im1, str) and isinstance(im2, str):
            if src_space != 'MNI152' and trg_space == 'MNI152':
                hemi = HEMI.search(im1).group(1)
            elif src_space == 'MNI152' and trg_space != 'MNI152':
                hemi = HEMI.search(im2).group(1)

        # resample images
        im1, im2 = resampling.resample_images(im1, im2, src_space, trg_space,
                                              hemi=hemi, alt_spec=alt_spec,
                                              resampling='transform_to_alt')

        # generate nulls maps
        spins = nullmaps
        if hemi == 'L':
            spins = np.split(spins, 2)[0]
            im1 = (im1[0],) if len(im1) == 2 else im1
            im2 = (im2[0],) if len(im2) == 2 else im2
        elif hemi == 'R':
            spins = np.split(spins, 2)[1] - len(images.load_data(im1))
            im1 = (im1[1],) if len(im1) == 2 else im1
            im2 = (im2[1],) if len(im2) == 2 else im2
        resampled = nulls.alexander_bloch(images.load_data(im1), *alt_spec,
                                          n_perm=1000, seed=1234, spins=spins)

        # correlate images
        corr, pval = stats.compare_images(im1, im2, nulls=resampled)
        df.append([annot1, annot2, corr, pval])
        print(df[-1])

    return pd.DataFrame(df, columns=['annot1', 'annot2', 'corr', 'pval'])


def main():
    # fetch relevant annotations
    annotations = ANNOTATIONS.copy()
    for i, img in enumerate(annotations):
        keys = (
            'source', 'desc', 'space', 'den' if img[2] != 'MNI152' else 'res'
        )
        annotations[i] = datasets.fetch_annotation(**dict(zip(keys, img)))[img]

    # transform annotations to every combination of atlas / density (exclude
    # MNI152 for now)
    for atlas, densities in datasets.DENSITIES.items():
        if atlas == 'MNI152':
            continue
        for den in densities:
            if atlas == 'civet' and den == '164k':
                continue
            if atlas == 'fsLR' and den in ['4k', '8k']:
                continue
            outdir = OUTDIR / atlas / den
            outdir.mkdir(exist_ok=True, parents=True)

            # checkpoint whether correlations have already been generated
            fn = outdir / 'correlations.csv'
            if fn.exists():
                continue

            # generate resampling arrays; load from disk if we can!
            if (outdir / 'spins.npy').exists():
                spins = np.load(outdir / 'spins.npy')
            else:
                spins = nulls.alexander_bloch(None, atlas, den,
                                              n_perm=1000, seed=1234)
                np.save(outdir / 'spins.npy', spins)

            # make pairwise correlations and save to CSV
            data = correlate_images(annotations, (atlas, den),
                                    nullmaps=spins)
            data.to_csv(fn, sep=',', index=False)


if __name__ == "__main__":
    main()
