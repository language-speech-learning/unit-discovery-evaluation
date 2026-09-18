from typing import Collection, Tuple

import numpy as np
import itertools
import statistics
import editdistance

from utils.utils import *


def distance(
    group: Collection[Union[str, int]], 
    total_dist: float, 
    total_count: int
) -> Tuple[Tuple[float, int], float, int]:
    """Computes the average pairwise normalized edit distance for a group of
    transcriptions (forward metrics) or cluster sequences (inverse metrics).
    Also keeps track of the total pairwise normalized edit distances and the 
    number of pairwise comparisons made to compute the original NED.

    Parameters
    ----------
    group : Collection of str or int
        The unit transcriptions or cluster sequences to compare using pairwise
        edit distance.
    total_dist : float
        The running total of the pairwise normalized edit distances across all 
        groups.
    total_count : int
        The running total of the number of pairwise comparisons made across all
        groups.

    Returns
    -------
    group_result : tuple of (float, int)
        The average pairwise normalized edit distances for this group and the 
        number of elements in the group (used for weighted metrics).
    total_dist : float
        The updated running total of the pairwise normalized edit distances.
    total_count : int
        The updated running total of the number of pairwise comparisons made.
    """

    num_elements = len(group)
    if num_elements == 1: # Singleton handling
        return (0.0, num_elements), total_dist, total_count

    group_dist = 0.0
    group_count = 0
    for p, q in itertools.combinations(group, 2):
        if isinstance(p, int) and isinstance(q, int): # binary match check
            dist = 0 if p == q else 1
        else: # edit distance calculation
            max_len = max(len(p), len(q))
            dist = editdistance.eval(p, q) / max_len if max_len > 0 else 1.0
        
        group_dist += dist
        group_count += 1
    
    total_dist += group_dist
    total_count += group_count

    return (group_dist/group_count, num_elements), total_dist, total_count


def ned(
    discovered: List[Tuple[Fragment, Transcription, int]]
) -> Tuple[float, float, float]:
    """Computes original, per-cluster normalized, and weighted NED metrics for 
    a set of discovered units grouped by cluster.

    Parameters
    ----------
    discovered : list of tuple of (Fragment, Transcription, int)
        Discovered fragments each containing a ``Fragment``, its 
        ``Transcription``, and its cluster ID.

    Returns
    -------
    NED : float
        Original NED. Not normalized (each cluster's weight is proportional to
        its # pairwise comparisons). Singleton clusters are ignored.
    Per-cluster norm NED : float
        Per-cluster normalized NED. Normalized such that each has equal weight
        regardless of its size. Singleton clusters are ignored.
    WNED : float
        Weighted NED. Normalized per number of elements in each cluster such 
        that each cluster's weight is proportional to its size. Singleton
        clusters are credited.
    """

    discovered = sorted(discovered, key=lambda x: x[2])

    tot_dist = 0.0
    tot_count = 0
    per_cluster_mean, weights, weighted_dist = [], [], []
    for _, group in itertools.groupby(discovered, key=lambda x: x[2]):
        (group_mean, group_size), tot_dist, tot_count = distance(
            [
                c[1].tokens for c in group
            ], 
            total_dist=tot_dist, 
            total_count=tot_count
        )
        
        weights.append(group_size)
        weighted_dist.append(group_mean)
        if group_size > 1: # Pop singleton clusters
            per_cluster_mean.append(group_mean)

    # NED, Per cluster norm NED, Weighted NED
    return tot_dist/tot_count, np.mean(per_cluster_mean), np.average(weighted_dist, weights=weights)


def inverse_ned(
    gt_unit_dict: Dict[str, List[Tuple[str, List[int], List[Interval]]]]
) -> Tuple[float, float, float]:
    """Computes (our) original, per-class normalized, and weighted inverse NED 
    metrics for a set of cluster sequences grouped by their underlying class.

    Parameters
    ----------
    gt_unit_dict : dict of {str: list of tuple}
        Keyed by gold unit class. Each value is a list of tuples containing
        the speaker, the list of overlapping cluster IDs, and the list of
        corresponding discovered ``Interval`` objects.

    Returns
    -------
    iNED : float
        Original inverse NED. Not normalized (each class's weight is 
        proportional to its # pairwise comparisons). Singleton classes are 
        ignored.
    Per-class norm iNED : float
        Per-class normalized inverse NED. Normalized such that each has equal 
        weight regardless of its size. Singleton classes are ignored.
    iWNED : float
        Weighted inverse NED. Normalized per number of elements in each class 
        such that each class's weight is proportional to its size. Singleton
        classes are ignored.
    """

    tot_dist = 0.0
    tot_count = 0
    per_class_mean, weights = [], []
    for clusters in gt_unit_dict:
        (group_mean, group_size), tot_dist, tot_count = distance(
            [
                tuple(item[1]) 
                for item in gt_unit_dict[clusters]
            ],
            total_dist=tot_dist, 
            total_count=tot_count
        )
        if group_size > 1: # Pop singleton classes
            weights.append(group_size)
            per_class_mean.append(group_mean)
    
    # iNED, Per class norm iNED, Weighted iNED
    return tot_dist/tot_count, np.mean(per_class_mean), np.average(per_class_mean, weights=weights)


def disc_type_ined(
    discovered: List[Tuple[Fragment, Transcription, int]]
) -> Tuple[float, float]:
    """Computes (our) original, and weighted inverse discovered-class NED 
    metrics for a set of cluster sequences grouped by their underlying 
    discovered class.

    Parameters
    ----------
    discovered : list of tuple of (Fragment, Transcription, int)
        Discovered fragments each containing a ``Fragment``, its 
        ``Transcription``, and its cluster ID.

    Returns
    -------
    iNED : float
        Original inverse NED. Not normalized (each discovered-class's weight is 
        proportional to its # pairwise comparisons). Singleton 
        discovered-classes are ignored.
    iWNED : float
        Weighted inverse NED. Normalized per number of elements in each 
        discovered-class such that each discovered-class's weight is 
        proportional to its size. Singleton discovered-classes are ignored.
    """

    discovered = sorted(discovered, key=lambda x: x[1].tokens)

    tot_dist = 0.0
    tot_count = 0
    weights, weighted_dist = [], []
    for _, group in itertools.groupby(discovered, key=lambda x: x[1].tokens):
        (group_mean, group_size), tot_dist, tot_count = distance([c[2] for c in group], 
                                                                total_dist=tot_dist, 
                                                                total_count=tot_count)
        if group_size > 1: # Pop singleton discovered classes
            weights.append(group_size)
            weighted_dist.append(group_mean)

    # Discovered-class iNED, Weighted discovered-class iNED
    return tot_dist/tot_count, np.average(weighted_dist, weights=weights)


def phone_edit_distance(
    group: Collection[Union[str, int]], 
    inverse: bool = False
) -> float:
    """Computes the average error rate between each item in the group and the 
    modal item of the group. Each group contains transcriptions (forward 
    metrics) or cluster sequences (inverse metrics).

    Parameters
    ----------
    group : Collection of str or int
        The unit transcriptions or cluster sequences to compare using pairwise
        edit distance.
    inverse : bool
        Flag to ensure correct singleton handling. ``False`` for forward, and 
        ``True`` for inverse metrics.
    Returns
    -------
    best_distances : float

    """
    
    if len(group) == 1:
        # Pop singleton class for inverse, credit singleton cluster for forward
        return [np.nan] if inverse else [0.0]
    
    modes = statistics.multimode(group)
    if len(modes) == 1: # Choose the single modal item
        modal_g = modes[0]
        length = len(modal_g)
        assert length > 0
        return [editdistance.eval(i, modal_g) / length for i in group]
    
    max_len = len(max(modes, key=len))
    longest_modes = [m for m in modes if len(m) == max_len]
    longest_modes_len = len(longest_modes[0])
    assert longest_modes_len > 0

    min_total_dist = float('inf')
    for candidate_modal_class in longest_modes:
        # Choose the modal item that is the longest and gives the lowest edit distance
        current_distances = [
            editdistance.eval(i, candidate_modal_class) / longest_modes_len 
            for i in group
        ]
        current_total = np.sum(current_distances)
        
        if current_total < min_total_dist:
            min_total_dist = current_total
            best_distances = current_distances
    
    return best_distances


def per(discovered: List[Tuple[Fragment, Transcription, int]]) -> float:
    """Computes the phone error rate metric for a set of transcriptions grouped 
    by their cluster ID.

    Parameters
    ----------
    discovered : list of tuple of (Fragment, Transcription, int)
        Discovered fragments each containing a ``Fragment``, its 
        ``Transcription``, and its cluster ID.

    Returns
    -------
    PER : float
        Phone error rate. Singleton clusters are credited.
    """

    discovered = sorted(discovered, key=lambda x: x[2])
    distances = [
        d
        for _, group in itertools.groupby(discovered, key=lambda x: x[2])
        for d in phone_edit_distance([c[1].tokens for c in group])
    ]
    return np.mean(distances) if len(distances) > 0 else 0.0


def inverse_per(
    gt_unit_dict: Dict[str, List[Tuple[str, List[int], List[Interval]]]], 
    inverse: bool = True
) -> float:
    """Computes the inverse phone error rate metric for a set of cluster 
    sequences grouped by their underlying class.

    Parameters
    ----------
    gt_unit_dict : dict of {str: list of tuple}
        Keyed by gold unit class. Each value is a list of tuples containing
        the speaker, the list of overlapping cluster IDs, and the list of
        corresponding discovered ``Interval`` objects.
    inverse : bool
        Flag to ensure correct singleton handling. Always set to ``True`` for 
        inverse.

    Returns
    -------
    iPER : float
        Inverse phone error rate. Singleton classes are ignored.
    """
    
    distances = [
        d
        for clusters in gt_unit_dict
        for d in phone_edit_distance(
            [tuple(p[1]) for p in gt_unit_dict[clusters]], 
            inverse=inverse
        )
    ]
    return np.nanmean(distances)