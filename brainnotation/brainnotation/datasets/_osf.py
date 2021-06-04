# -*- coding: utf-8 -*-
"""
Functions for working with data on OSF
"""

from brainnotation.datasets.utils import _get_session


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
