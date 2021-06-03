# -*- coding: utf-8 -*-
"""
Functions for fetching annotations (from the internet, if necessary)
"""

from collections import defaultdict
import os
import re

from nilearn.datasets.utils import _fetch_files
import requests

from brainnotation.datasets.utils import get_data_dir, get_dataset_info

MATCH = re.compile(
    r'source-(\S+)_desc-(\S+)_space-(\S+)_(?:den|res)-(\d+[k|m]{1,2})_'
)


def available_annotations(source=None, desc=None, space=None, den=None,
                          res=None, hemi=None, tags=None):
    """
    Lists datasets available via :func:`~.fetch_annotation`

    Parameters
    ----------
    source, desc, space, den, res, hemi, tags : str or list-of-str
        Values on which to match annotations. If not specified annotations with
        any value for the relevant key will be matched. Default: None

    Returns
    -------
    datasets : list-of-str or dict
        List of available annotations. If `return_description` is True, a dict
        is returned instead where keys are available annotations and values are
        brief descriptions of the annotation.
    """

    info = get_dataset_info('annotations')

    return _match_annot(info, source=source, desc=desc, space=space, den=den,
                        res=res, hemi=hemi, tags=tags)


def _match_annot(info, **kwargs):
    """
    Matches datasets in `info` to relevant keys

    Parameters
    ----------
    info : list-of-dict
        Information on annotations

    Returns
    -------
    matched : list-of-dict
        Annotations with specified values for keys
    """

    # tags should always be a list
    tags = kwargs.get('tags')
    if tags is not None and isinstance(tags, str):
        kwargs['tags'] = [tags]

    # 'den' and 'res' are a special case because these are mutually exclusive
    # values (only one will ever be set for a given annotation) so we want to
    # match on _either_, not both, if and only if both are provided as keys.
    # if only one is specified as a key then we should exclude the other!
    denres = []
    for vals in (kwargs.get('den'), kwargs.get('res')):
        vals = [vals] if isinstance(vals, str) else vals
        if vals is not None:
            denres.extend(vals)

    out = []
    for dset in info:
        match = (dset.get('den') or dset.get('res')) in denres
        for key in ('source', 'desc', 'space', 'hemi', 'tags'):
            comp, value = dset.get(key), kwargs.get(key)
            if value is None:
                continue
            elif value is not None and comp is None:
                match = False
            elif isinstance(value, str):
                match = match and comp == value
            else:
                func = all if key == 'tags' else any
                match = match and func(f in comp for f in value)
        if match:
            out.append(dset)

    return out


def fetch_annotation(*, source=None, desc=None, space=None, den=None, res=None,
                     hemi=None, tags=None, token=None, data_dir=None,
                     verbose=1):
    """
    Downloads files for brain annotations matching requested variables

    Parameters
    ----------
    source, desc, space, den, res, hemi, tags : str or list-of-str
        Values on which to match annotations. If not specified annotations with
        any value for the relevant key will be matched. Default: None
    token : str, optional
        OSF personal access token for accessing restricted annotations. Will
        also check the environmental variable 'BRAINNOTATION_OSF_TOKEN' if not
        provided; if that is not set no token will be provided and restricted
        annotations will be inaccessible. Default: None
    data_dir : str, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'BRAINNOTATION_DATA'; if that is not set, will
        use `~/brainnotation-data` instead. Default: None
    verbose : int, optional
        Modifies verbosity of download, where higher numbers mean more updates.
        Default: 1

    Returns
    -------
    data : dict
        Dictionary of downloaded annotations where dictionary keys are tuples
        (source, desc, space, den/res) and values are lists of corresponding
        filenames
    """

    data_dir = get_data_dir(data_dir=data_dir)
    info = _match_annot(get_dataset_info('annotations'),
                        source=source, desc=desc, space=space, den=den,
                        res=res, hemi=hemi, tags=tags)
    if verbose:
        print(f'Identified {len(info)} datsets matching specified parameters')

    # get and add token to headers if present
    headers, session = {}, None
    if token is None:
        token = os.environ.get('BRAINNOTATION_OSF_TOKEN', None)
    if token is not None:
        headers['Authorization'] = 'Bearer {}'.format(token)
        session = requests.Session()
        session.headers.update(headers)

    filenames = [
        (os.path.join('annotations', dset['fname']),
         dset['url'],
         {'md5sum': dset['checksum']}) for dset in info
    ]
    data = _fetch_files(data_dir, files=filenames, verbose=verbose,
                        session=session)

    out = defaultdict(list)
    for fn in data:
        out[MATCH.search(fn).groups()].append(fn)

    return {k: v for k, v in out.items()}
