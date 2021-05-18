# -*- coding: utf-8 -*-

import numpy as np


def point_in_triangle(point, triangle, return_pdist=True):
    """
    Checks whether `point` falls inside `triangle`

    Parameters
    ----------
    point : (3,) array_like
        Coordinates of point
    triangle (3, 3) array_like
        Coordinates of triangle
    return_pdist : bool, optional
        Whether to return planar distance (see outputs). Default: True

    Returns
    -------
    inside : bool
        Whether `point` is inside triangle
    pdist : float
        The approximate distance of the point to the plane of the triangle.
        Only returned if `return_pdist` is True
    """

    A, B, C = triangle
    v0 = C - A
    v1 = B - A
    v2 = point - A

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    denom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * denom
    v = (dot00 * dot12 - dot01 * dot02) * denom
    inside = (u >= 0) and (v >= 0) and (u + v < 1)

    if return_pdist:
        return inside, np.abs(v2 @ np.cross(v1, v0))

    return inside


def which_triangle(point, triangles):
    """
    Determines which of `triangles` the provided `point` falls inside

    Parameters
    ----------
    point : (3,) array_like
        Coordinates of point
    triangles : (N, 3, 3) array_like
        Coordinates of `N` triangles to check

    Returns
    -------
    idx : int
        Index of `triangles` that `point` is inside of. If `point` does not
        fall within any of `triangles` then this will be None
    """

    idx, planar = None, np.inf
    for n, tri in enumerate(triangles):
        inside, pdist = point_in_triangle(point, tri)
        if pdist < planar and inside:
            idx, planar = n, pdist

    return idx


def get_shared_triangles(faces):
    """
    Returns dictionary of triangles sharing edges from `faces`

    Parameters
    ----------
    faces : (N, 3)
        Triangles comprising mesh

    Returns
    -------
    shared : dict
        Where keys are len-2 tuple of vertex ids for the shared edge and values
        are the triangles that have this shared edge.
    """

    faces = np.asarray(faces)
    edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2)), axis=1)
    edges_face = np.repeat(np.arange(len(faces)), 3)

    order = np.lexsort(edges.T[::-1])
    edges_sorted = edges[order]
    dupe = np.any(edges_sorted[1:] != edges_sorted[:-1], axis=1)
    dupe_idx = np.append(0, np.nonzero(dupe)[0] + 1)
    start_ok = np.diff(np.concatenate((dupe_idx, [len(edges_sorted)]))) == 2
    groups = np.tile(dupe_idx[start_ok].reshape(-1, 1), 2)
    edge_groups = order[groups + np.arange(2)]

    adjacency = edges_face[edge_groups]
    nondegenerate = adjacency[:, 0] != adjacency[:, 1]
    adjacency = np.sort(adjacency[nondegenerate], axis=1)
    adjacency_edges = edges[edge_groups[:, 0][nondegenerate]]

    indirect_edges = np.zeros(adjacency.shape, dtype=np.int32) - 1

    for i, fid in enumerate(adjacency.T):
        face = faces[fid]
        unshared = np.logical_not(np.logical_or(
            face == adjacency_edges[:, 0].reshape(-1, 1),
            face == adjacency_edges[:, 1].reshape(-1, 1)))
        row_ok = unshared.sum(axis=1) == 1
        unshared[~row_ok, :] = False
        indirect_edges[row_ok, i] = face[unshared]

    shared = np.sort(face[np.logical_not(unshared)].reshape(-1, 1, 2), axis=-1)
    shared = np.repeat(shared, 2, axis=1)
    triangles = np.concatenate((shared, indirect_edges[..., None]), axis=-1)

    return dict(zip(map(tuple, adjacency_edges), triangles))
