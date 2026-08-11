from typing import Iterable, Tuple

import numpy as np
import itertools
import statistics
import editdistance

from utils.utils import *


def distance(group: Iterable[Tuple[str]], total_dist, total_count) -> float:
    num_elements = len(group)
    if num_elements == 1: # pop singleton classes for reverse metric
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


def inverse_ned(gt_word_dict) -> float:
    """
    (Original) R-NED:
        Not normalized (each class' weight is proportional to its # pairwise comparisons)
        Pop singleton classes
    Per-cluster Norm R-NED:
        Normalized per class (each class has equal weight)
        Pop singleton classes
    Per-cluster # elements Norm R-NED:
        Normalized per # elements in class (each class' weight is proportional to its # elements)
        Pop singleton classes
    """

    tot_dist = 0.0
    tot_count = 0
    per_cluster_mean, weights, weighted_dist = [], [], []
    for clusters in gt_word_dict:
        (group_mean, group_size), tot_dist, tot_count = distance(
            [
                tuple(item[1]) 
                for item in gt_word_dict[clusters]
            ],
            total_dist=tot_dist, 
            total_count=tot_count
        )
        if group_size > 1:
            weights.append(group_size)
            weighted_dist.append(group_mean)
            per_cluster_mean.append(group_mean)
    
    # Original R-NED, Per cluster weighted R-NED, Per # elements weighted R-NED
    return tot_dist/tot_count, np.mean(per_cluster_mean), np.average(weighted_dist, weights=weights)


def disc_type_ined(discovered: Iterable[Tuple[Fragment, Transcription, int]]) -> float:
    """
    Original discovered-type iNED:
        Not normalized (each cluster's weight is proportional to its # pairwise comparisons)
        Pop singleton clusters
    Weighted discovered-type iNED:
        Normalized per # elements in cluster (each cluster's weight is proportional to its # elements)
        Credit singleton clusters
    """
    discovered = sorted(discovered, key=lambda x: x[1].tokens)

    tot_dist = 0.0
    tot_count = 0
    weights, weighted_dist = [], []
    for _, group in itertools.groupby(discovered, key=lambda x: x[1].tokens):
        (group_mean, group_size), tot_dist, tot_count = distance([c[2] for c in group], 
                                                                total_dist=tot_dist, 
                                                                total_count=tot_count)
        if group_size > 1: # pop singleton discovered types
            weights.append(group_size)
            weighted_dist.append(group_mean)

    # Original discovered-type iNED, Weighted discovered-type iNED
    return tot_dist/tot_count, np.average(weighted_dist, weights=weights)


def ned(discovered: Iterable[Tuple[Fragment, Transcription, int]]) -> float:
    """
    Original NED:
        Not normalized (each cluster's weight is proportional to its # pairwise comparisons)
        Pop singleton clusters
    Per-cluster Norm NED:
        Normalized per cluster (each cluster has equal weight)
        Pop singleton clusters
    Per-cluster # elements Norm NED:
        Normalized per # elements in cluster (each cluster's weight is proportional to its # elements)
        Credit singleton clusters
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
        if group_size > 1: per_cluster_mean.append(group_mean)

    # Original NED, Per cluster weighted NED, Per # elements weighted NED
    return tot_dist/tot_count, np.mean(per_cluster_mean), np.average(weighted_dist, weights=weights)


def phone_edit_distance(group: Iterable[Tuple[str]], reverse: bool = False) -> float:
    if len(group) == 1:
        return [np.nan] if reverse else [0.0] # pop singleton class for reverse metric, credit singleton cluster for forward metric
    
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
    for candidate_modal_class in longest_modes: # Choose the modal item that is the longest and gives the lowest edit 
        current_distances = [
            editdistance.eval(i, candidate_modal_class) / longest_modes_len 
            for i in group
        ]
        current_total = np.sum(current_distances)
        
        if current_total < min_total_dist:
            min_total_dist = current_total
            best_distances = current_distances
    
    return best_distances


def inverse_per(gt_word_dict, reverse: bool = True) -> float:
    """
    Not normalized (average over all comparisons)
    Pop singleton classes
    """
    distances = [
        d
        for clusters in gt_word_dict
        for d in phone_edit_distance(
            [tuple(p[1]) for p in gt_word_dict[clusters]], 
            reverse=reverse
        )
    ]
    return np.nanmean(distances)


def per(discovered: Iterable[Tuple[Fragment, Transcription, int]]) -> float:
    """
    Not normalized (average over all comparisons)
    Credit singleton clusters
    """

    discovered = sorted(discovered, key=lambda x: x[2])
    distances = [
        d
        for _, group in itertools.groupby(discovered, key=lambda x: x[2])
        for d in phone_edit_distance([c[1].tokens for c in group])
    ]
    return np.mean(distances) if len(distances) > 0 else 0.0