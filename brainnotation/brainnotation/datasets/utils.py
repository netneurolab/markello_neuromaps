# -*- coding: utf-8 -*-
"""
Utilites for loading / creating datasets
"""

import json
import os
from pkg_resources import resource_filename


def _osfify_urls(data):
    """
    Formats `data` object with OSF API URL

    Parameters
    ----------
    data : object
        If dict with a `url` key, will format OSF_API with relevant values

    Returns
    -------
    data : object
        Input data with all `url` dict keys formatted
    """

    OSF_API = "https://files.osf.io/v1/resources/{}/providers/osfstorage/{}"

    if isinstance(data, str) or data is None:
        return data
    elif 'url' in data:
        # if url is None then we this is a malformed entry and we should ignore
        if data['url'] is None:
            return
        # if the url isn't a string assume we're supposed to format it
        elif not isinstance(data['url'], str):
            data['url'] = OSF_API.format(*data['url'])

    try:
        for key, value in data.items():
            data[key] = _osfify_urls(value)
    except AttributeError:
        for n, value in enumerate(data):
            data[n] = _osfify_urls(value)
        # drop the invalid entries
        data = [d for d in data if d is not None]

    return data


def get_dataset_info(name):
    """
    Returns url and MD5 checksum for dataset `name`

    Parameters
    ----------
    name : str
        Name of dataset
    dtype : {'atlas', 'annotation'}, optional
        What datatype `name` refers to. Default: 'atlas'

    Returns
    -------
    url : str
        URL from which to download dataset
    md5 : str
        MD5 checksum for file downloade from `url`
    """

    with open(resource_filename('brainnotation', 'data/osf.json')) as src:
        osf_resources = _osfify_urls(json.load(src))

    try:
        return osf_resources[name]
    except KeyError:
        raise KeyError("Provided dataset '{}' is not valid. Must be one of: {}"
                       .format(name, sorted(osf_resources.keys())))


def get_data_dir(data_dir=None):
    """
    Gets path to brainnotation data directory

    Parameters
    ----------
    data_dir : str, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'BRAINNOTATION_DATA'; if that is not set, will
        use `~/brainnotation-data` instead. Default: None

    Returns
    -------
    data_dir : str
        Path to use as data directory
    """

    if data_dir is None:
        data_dir = os.environ.get('BRAINNOTATION_DATA',
                                  os.path.join('~', 'brainnotation-data'))
    data_dir = os.path.expanduser(data_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    return data_dir
