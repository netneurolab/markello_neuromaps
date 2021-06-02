"""
Functions for fetching datasets
"""

__all__ = [
    'fetch_all_atlases', 'fetch_atlas', 'fetch_civet', 'fetch_fsaverage',
    'fetch_fslr', 'fetch_mni152', 'fetch_regfusion', 'get_atlas_dir',
    'DENSITIES', 'ALIAS',
]

from .fetchers import (fetch_all_atlases, fetch_atlas, fetch_civet,
                       fetch_fsaverage, fetch_fslr, fetch_mni152,
                       fetch_regfusion, get_atlas_dir, DENSITIES, ALIAS)
