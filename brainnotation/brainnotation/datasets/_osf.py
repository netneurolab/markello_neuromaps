# -*- coding: utf-8 -*-
"""
Functions for working with data/osf.json file
"""

import os
from pkg_resources import resource_filename
import json

from nilearn.datasets.utils import _md5_sum_file

from brainnotation.datasets.utils import _get_session

# uniquely identify each item ('hemi' can be None)
FNAME_KEYS = ['source', 'desc', 'space', 'den', 'res', 'hemi']
# auto-generated (checksum can be None if file doest not exist)
AUTO_KEYS = ['format', 'fname', 'rel_path', 'checksum']
# required keys but values are all optional
COND_KEYS = ['title', 'tags', 'redir', 'url']
# minimal keys for each item
MINIMAL_KEYS = FNAME_KEYS + AUTO_KEYS + COND_KEYS
# keys for redirection
REDIR_KEYS = ['space', 'den']
# keys for more metadata (unique for each source)
INFO_KEYS = ['source', 'refs', 'comments', 'demographics']

# distribution JSON
OSFJSON = resource_filename('brainnotation', 'data/osf.json')


def parse_filename(fname, return_ext=True, verbose=False):
    """
    Parses `fname` (in BIDS-inspired format) and returns dictionary

    Parameters
    ----------
    fname : str os os.PathLike
        Filename to parse
    return_ext : bool, optional
        Whether to return extension of `fname` in addition to key-value dict.
        Default: False
    verbose : bool, optional
        Whether to print status messages. Default: False

    Returns
    -------
    info : dict
        Key-value pairs extracted from `fname`
    ext : str
        Extension of `fname`, only returned if `return_ext=True`
    """

    try:
        base, *ext = fname.split('.')
        fname_dict = dict([
            pair.split('-') for pair in base.split('_') if pair != 'feature'
        ])
    except ValueError:
        print('Wrong filename format!')
        return

    if verbose:
        print(fname_dict)

    if return_ext:
        return fname_dict, '.'.join(ext)

    return fname_dict


def parse_fname_list(fname, verbose=False):
    """
    Reads in list of BIDS-inspired filenames from `fname` and parses keys

    Parameters
    ----------
    fname : str os os.PathLike
    verbose : bool, optional
        Whether to print status messages. Default: False

    Returns
    -------
    data : list-of-dict
        Information about filenames in `fname`
    """

    with open(fname, 'r', encoding='utf-8') as src:
        fname_list = [name.strip() for name in src.readlines()]
    data = [
        parse_filename(name, return_ext=False, verbose=verbose)
        for name in fname_list
    ]
    if verbose:
        print(fname_list)

    return data


def parse_json(fname, root='annotations'):
    """
    Loads JSON from `fname` and returns value of `root` key(s)

    Parameters
    ----------
    fname : str or os.PathLike
        Filepath to JSON file
    root : str or list-of-str, optional
        Root key(s) to query JSON file. Default: 'annotations'

    Returns
    -------
    data : dict
        Data from `fname` JSON file
    """

    if isinstance(root, str):
        root = [root]

    with open(fname, 'r', encoding='utf-8') as src:
        data = json.load(src)

    for key in root:
        data = data[key]

    return data


def write_json(data, fname, root='annotations', indent=4):
    """
    Saves `data` to `fname` JSON

    Parameters
    ----------
    data : JSON-compatible format
        Data to save to `fname`
    fname : str or os.PathLike
        Path to filename where `data` should be saved as JSON
    root : str, optional
        Key to save `data` in `fname`. Default: 'annotations'
    indent : int, optional
        Indentation of JSON file. Default: 4

    Returns
    -------
    fname : str
        Path to saved file
    """

    if not isinstance(root, str):
        raise ValueError(f'Provided `root` must be a str. Received: {root}')

    # if `fname` already exists we want to update it, not overwrite it!
    if os.path.isfile(fname):
        output = parse_json(fname, [])
    output[root] = data

    # save to disk
    with open(fname, 'w', encoding='utf-8') as dest:
        json.dump(output, dest, indent=indent)

    return fname


def complete_json(input_data, ref_keys='minimal', input_root=None,
                  output_fname=None, output_root=None):
    """
    Parameters
    ----------
    input_data : str or os.PathLike or list-of-dict
        Filepath to JSON with data or list of dictionaries with information
        about annotations
    ref_keys : {'minimal', 'info'}, optional
        Which reference keys to check in `input_data`. Default: 'minimal'
    input_root : str, optional
        If `input_data` is a filename the key in the file containing data about
        annotations. If not specified will be based on provided `ref_keys`.
        Default: None
    output_fname : str or os.PathLike, optional
        Filepath where complete JSON should be saved. If not specified the
        data are not saved to disk. Default: None
    output_root : str, optional
        If `output_fname` is not None, the key in the saved JSON where
        completed information should be stored. If not specified will be based
        on `input_root`. Default: None

    Returns
    -------
    output : list-of-dict
        Information about annotations from `input_data`
    """
    valid_keys = ['minimal', 'info']
    if ref_keys not in valid_keys:
        raise ValueError(f'Invalid ref_keys: {ref_keys}. Must be one of '
                         f'{valid_keys}')

    # this is to add missing fields to existing data
    # could accept data dict list or filename as input
    # set minimal vs info
    if ref_keys == 'minimal':
        ref_keys = MINIMAL_KEYS
        if input_root is None:
            input_root = 'annotations'
    elif ref_keys == 'info':
        ref_keys = INFO_KEYS
        if input_root is None:
            input_root = 'info'

    # check input
    if not isinstance(input_data, list):
        input_data = parse_json(input_data, root=input_root)

    # make output
    output = []
    for item in input_data:
        output.append({
            key: (item[key] if key in item else None)
            for key in ref_keys
        })

    # write output
    if output_fname is not None:
        if output_root is None:
            output_root = input_root
        write_json(output, output_fname, root=output_root)

    return output


def check_missing_keys(fname, root='annotations'):
    """
    Checks whether data in `fname` JSON are missing required keys

    Required keys are specified in ``brainnotation.datasets._osf.MINIMAL_KEYS``

    Parameters
    ----------
    fname : str or os.PathLike
        Filepath to JSON file to check
    root : str or list-of-str, optional
        Root key(s) to query JSON file. Default: 'annotations'

    Returns
    -------
    info : list of list-of-str
        Missing keys for each entry in `fname`
    """

    data = parse_json(fname, root=root)

    is_missing_keys, info = False, []
    for item in data:
        missing = sorted(set(MINIMAL_KEYS) - set(item))
        if len(missing) > 0:
            is_missing_keys = True
        info.append(missing)

    if is_missing_keys:
        raise KeyError('Data in provided `fname` are missing some keys. '
                       'Please use `brainnotation.datasets._osf.complete_json`'
                       ' to fill missing keys')

    return info


def generate_auto_keys(item):
    """
    Adds automatically-generated keys to `item`

    Generated keys include: ['format', 'fname', 'rel_path', 'checksum']

    Parameters
    ----------
    item : dict
        Information about annotation

    Returns
    -------
    item : dict
        Updated information about annotation
    """

    item = item.copy()

    pref = 'source-{source}_desc-{desc}_space-{space}'
    surffmt = pref + '_den-{den}_hemi-{hemi}_feature.func.gii'
    volfmt = pref = '_res-{res}_feature.nii.gz'

    # check format by checking 'hemi'
    is_surface = item['den'] or item['hemi'] or item['format'] == 'surface'
    is_volume = item['res'] or item['format'] == 'volume'

    if is_surface:  # this is surface file
        item['format'] = 'surface'
        item['fname'] = surffmt.format(**item)
    elif is_volume:  # this is volume file
        item['format'] = 'volume'
        item['fname'] = volfmt.format(**item)
    else:
        print('Missing keys to determine surface/volumetric format of data; '
              'fname keys not generated')

    item['rel_path'] = os.path.join(*[
        item[key] for key in ['source', 'desc', 'space']
    ])

    # check file existence
    filepath = os.path.join(item['rel_path'], item['fname'])
    if item['fname'] is not None and os.path.isfile(filepath):
        item['checksum'] = _md5_sum_file(filepath)

    return item


def clean_minimal_keys(item):
    """
    Removes incompatible keys from `item` based on `item['format']`

    Parameters
    ----------
    item : dict
        Information about annotation

    Returns
    -------
    item : dict
        Updated information about annotation
    """

    keys = {'surface': ['res'], 'volume': ['den', 'hemi']}
    fmt = item.get('format')
    if fmt is None:
        print('Invalid value for format key; setting to "null"')
        item['format'] = None
        return

    for key in keys.get(fmt, []):
        item.pop(key, None)

    return item


def get_url(fname, project, token=None):
    """
    Gets OSF API URL path for `fname` in `project`

    Parameters
    ----------
    fname : str
        Filepath as it exists on OSF
    project : str
        Project ID on OSF
    token : str, optional
        OSF personal access token for accessing restricted annotations. Will
        also check the environmental variable 'BRAINNOTATION_OSF_TOKEN' if not
        provided; if that is not set no token will be provided and restricted
        annotations will be inaccessible. Default: None

    Returns
    -------
    path : str
        Path to `fname` on OSF project `project`
    """

    url = f'https://files.osf.io/v1/resources/{project}/providers/osfstorage/'
    session = _get_session(token=token)
    path = ''
    for pathpart in fname.strip('/').split('/'):
        out = session.get(url + path)
        out.raise_for_status()
        for item in out.json()['data']:
            if item['attributes']['name'] == pathpart:
                break
        path = item['attributes']['path'][1:]

    return path


def generate_release_json(fname, output=OSFJSON, root='annotations',
                          project=None, token=None):
    """
    Generates distribution-ready JSON file for fetching annotation data

    Parameters
    ----------
    fname : str or os.PathLike
        Path to filename where manually-edited JSON information is stored
    output : str or os.PathLike
        Path to filename where output JSON should be saved
    root : str, optional
        Key in `fname` where relevant data are stored. Default: 'annotations'
    project : str, optional
        Project ID on OSF where data files are stored. If not specified then
        the URL for the generated data will not be set. Default: None
    token : str, optional
        OSF personal access token for accessing restricted annotations. Will
        also check the environmental variable 'BRAINNOTATION_OSF_TOKEN' if not
        provided; if that is not set no token will be provided and restricted
        annotations will be inaccessible. Default: None

    Returns
    -------
    output : str
        Path to filename where output JSON was saved
    """

    output = []
    for item in parse_json(fname, root=root):
        item = clean_minimal_keys(generate_auto_keys(item))
        # fetch URL for file if needed (and project is specified)
        if (item.get('fname') is not None and item.get('url') is None
                and project is not None):
            fn = os.path.join(item['rel_path'], item['fname'])
            item['url'] = [project, get_url(fn, project=project, token=token)]
        output.append({key: item[key] for key in MINIMAL_KEYS if key in item})
    fname = write_json(output, output, root='annotations')
    return fname
