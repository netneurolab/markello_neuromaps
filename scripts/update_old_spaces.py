#!/usr/bin/env python
"""
Script for converting data in "old" formats to newer standard

Two datasets to be converted:

    1. Hill et al., 2010, PNAS (pals-ta24 --> fsLR)
    2. Reardon, Seidlitz, et al., 2018, Science (civet v1 --> civet v2)

In both cases nearest neighbors interpolation is used (however, in the case of
Hill2010 we use the deformation maps provided by the WUSTL group).
"""

from itertools import product
from pathlib import Path

import nibabel as nib

from brainnotation import images
from brainnotation_dev import caret

ATLASDIR = Path('./data/raw/atlases').resolve()
DATADIR = Path('./data/raw/brain_maps').resolve()
ORIGDIR = DATADIR / 'orig'


def map_hill2010():
    """ Converts maps from Hill et al 2010 to fsLR 164k space
    """
    deform = ATLASDIR / 'fsLR' / 'palsb12' \
        / 'tpl-palsb12_space-fsLR_den-74k_hemi-R_deform.txt'
    fmt = 'source-hill2010_desc-{desc}_space-{space}_den-{den}_hemi-R_' \
        'feature.func.gii'
    for desc in ('devexp', 'evoexp'):
        img = ORIGDIR / fmt.format(desc=desc, space='palsta24', den='74k')
        out = caret.apply_deform_map(img, deform)
        outdir = DATADIR / 'hill2010' / desc / 'fsLR'
        outdir.mkdir(exist_ok=True, parents=True)
        nib.save(images.construct_shape_gii(out),
                 outdir / fmt.format(desc=desc, space='fsLR', den='164k'))


def map_reardon2018():
    """ Converts maps from Reardon, Seidlitz et al 2018 to civet v2 space
    """
    src = ATLASDIR / 'civet' / 'civet_v1' \
        / 'tpl-civetv1_den-41k_hemi-{hemi}_midthickness.surf.gii'
    trg = ATLASDIR / 'civet' \
        / 'tpl-civet_den-41k_hemi-{hemi}_midthickness.surf.gii'
    fmt = 'source-reardon2018_desc-scaling{desc}_space-{space}_den-41k_'\
        'hemi-{hemi}_feature.func.gii'
    for desc, hemi in product(('hcp', 'pnc', 'nih'), ('L', 'R')):
        s, t = str(src).format(hemi=hemi), str(trg).format(hemi=hemi)
        img = ORIGDIR / fmt.format(desc=desc, space='civetv1', hemi=hemi)
        out = images.interp_surface(img, s, t)
        outdir = DATADIR / 'reardon2018' / f'scaling{desc}' / 'civet'
        outdir.mkdir(exist_ok=True, parents=True)
        nib.save(images.construct_shape_gii(out),
                 outdir / fmt.format(desc=desc, space='civet', hemi=hemi))


if __name__ == "__main__":
    map_hill2010()
    map_reardon2018()
