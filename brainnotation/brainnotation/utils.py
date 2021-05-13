# -*- coding: utf-8 -*-
"""
Utility functions
"""

import os
from pathlib import Path
import tempfile


def tmpname(suffix, prefix=None, directory=None):
    """
    Little helper function because :man_shrugging:

    Parameters
    ----------
    suffix : str
        Suffix of created filename

    Returns
    -------
    fn : str
        Temporary filename; user is responsible for deletion
    """

    fd, fn = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=directory)
    os.close(fd)

    return Path(fn)
