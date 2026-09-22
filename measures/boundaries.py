from typing import List, Tuple, Union

import itertools
import numpy as np


def get_p_r_f1(n_seg: int, n_ref: int, n_hit: int) -> Tuple[float, float, float]:
    """Calculate boundary precision, recall, F-score.

    Parameters
    ----------
    n_seg : int
        The number of segmentation boundaries.
    n_ref : int
        The number of reference boundaries.
    n_hit : int
        The number of seg-ref hits.

    Returns
    -------
    output : (float, float, float)
        precision, recall, F-score.
    """

    # Calculate metrics, avoid division by zero:
    if n_seg == n_ref == 0:
        return 0, 0, -np.inf
    elif n_hit == 0:
        return 0, 0, 0
    
    if n_seg != 0:
        precision = float(n_hit/n_seg)
    else:
        precision = np.inf
    
    if n_ref != 0:
        recall = float(n_hit/n_ref)
    else:
        recall = np.inf
    
    if precision + recall != 0:
        f1_score = 2*precision*recall/(precision+recall)
    else:
        f1_score = -np.inf
    
    return precision, recall, f1_score

def get_os(n_seg: int, n_ref: int) -> float:
    """Calculates over-segmentation: how many fewer/more boundaries are proposed 
    compared to the ground-truth.

    Parameters
    ----------
    n_seg : int
        The number of segmentation boundaries.
    n_ref : int
        The number of reference boundaries.

    Returns
    -------
    output : float
        Over-segmentation
    """

    if n_ref == 0:
        return -np.inf
    else:
        return n_seg/n_ref - 1
    
def get_rvalue(os: float, recall: float) -> Tuple[float, float]:
    """Calculates the R-value: how close the segmentation is to an ideal point 
    of operation (100% Recall with 0% OS).

    Parameters
    ----------
    os : float
        How many fewer/more boundaries are proposed compared to the ground-truth.
    recall : float
        How often the reference boundaries are contained in the segmentation.

    Returns
    -------
    output : float
        R-Value
    """

    r1 = np.sqrt((1 - recall)**2 + os**2)
    r2 = (-os + recall - 1)/np.sqrt(2)

    return 1 - (np.abs(r1) + np.abs(r2))/2

def eval_boundaries(
    seg: List[List[Union[int, float]]], 
    ref: List[List[Union[int, float]]], 
    tolerance: Union[int, float], strict: bool = True, 
    n_seg: int = 0, n_ref: int = 0, n_hit: int = 0
) -> Tuple[int, int, int]:
    """Count number of seg-ref hits based on the `tolerance`.

    Parameters
    ----------
    seg : list of list of shape (n_utterances, )
        Segmentation hypothesis to evaluate.
    ref : list of list of shape (n_utterances, )
        Ground-truth segmentation used as reference.
    tolerance : numerical
        The number of frames or seconds within which a seg boundary can hit 
        a ref boundary. If `int`, interpreted as number of frames; if `float`, 
        interpreted as number of seconds.
    strict : bool, default=True
        If `True`, a reference boundary can only be hit once. If `False`, 
        a reference boundary can be hit multiple times.
    n_seg : int, default=0
        The number of segmentation to start from.
    n_ref : int, default=0
        The number of reference boundaries to start from.
    n_hit : int, default=0
        The number of hits to start from.

    Returns
    -------
    output : (int, int, int)
        n_seg, n_ref, n_hit
    """
    
    assert len(seg) == len(ref) # Check if the number of utterances in the hypothesis and reference are the same
    for i_utterance in range(len(seg)):
        prediction = list(seg[i_utterance])
        ground_truth = list(ref[i_utterance])

        if (
            len(prediction) > 0
            and len(ground_truth) > 0
            and abs(prediction[-1] - ground_truth[-1]) <= tolerance
        ): # if the last boundary is within the tolerance, delete it since it would have hit
            prediction = prediction[:-1]
            ground_truth = ground_truth[:-1]
        # this helps when the segmentation algo does not automatically predict a boundary at the end of the utterance

        n_seg += len(prediction)
        n_ref += len(ground_truth)

        if len(prediction) == 0 or len(ground_truth) == 0: # no hits possible
            continue

        # # hits
        for i_ref in ground_truth:
            for i, i_seg in enumerate(prediction):
                if abs(i_ref - i_seg) <= tolerance:
                    n_hit += 1
                    prediction.pop(i) # remove the segmentation boundary that was hit
                    if strict: break # makes the evaluation strict, so that a reference boundary can only be hit once

    return n_seg, n_ref, n_hit

def eval_token_boundaries(
    seg: List[List[Union[int, float]]], 
    ref: List[List[Union[int, float]]], 
    tolerance: Union[int, float], strict: bool = True, 
    n_tokens_seg: int = 0, n_tokens_ref: int = 0, n_tokens_hit: int = 0
) -> Tuple[int, int, int]:
    """Count number of token (onset-offset) seg-ref hits.

    Parameters
    ----------
    seg : list of list of shape (n_utterances, )
        Segmentation hypothesis to evaluate.
    ref : list of list of shape (n_utterances, )
        Ground-truth segmentation used as reference.
    tolerance : numerical
        The number of frames or seconds within which a seg boundary can hit 
        a ref boundary. If `int`, interpreted as number of frames; if `float`, 
        interpreted as number of seconds.
    strict : bool, default=True
        If `True`, a reference boundary can only be hit once. If `False`, a 
        reference boundary can be hit multiple times.
    n_tokens_seg : int, default=0
        The number of segmentation to start from.
    n_tokens_ref : int, default=0
        The number of reference boundaries to start from.
    n_tokens_hit : int, default=0
        The number of hits to start from.

    Returns
    -------
    output : (int, int, int)
        n_tokens_seg, n_tokens_ref, n_tokens_hit
    """

    assert len(seg) == len(ref)
    for i_utterance in range(len(seg)): # for each utterance
        prediction = list(seg[i_utterance])
        ground_truth = list(ref[i_utterance])

        seg_segments = [(a,b) for a,b in itertools.pairwise([0] + prediction)]
        ref_segments = [(a,b) for a,b in itertools.pairwise([0] + ground_truth)]

        # Build list of ((ref_start_lower, ref_start_upper), (ref_end_lower, ref_end_upper))
        ref_intervals = []
        for word_start, word_end in ref_segments:
            ref_intervals.append(
                (
                    (max(0, word_start - tolerance), word_start + tolerance),
                    (word_end - tolerance, word_end + tolerance)
                )
            )
        
        n_tokens_ref += len(ref_intervals)
        n_tokens_seg += len(seg_segments)

        # Score word token boundaries
        for seg_start, seg_end in seg_segments:
            for i_gt_word, (ref_start_interval, ref_end_interval) in enumerate(
                ref_intervals
            ):
                ref_start_lower, ref_start_upper = ref_start_interval
                ref_end_lower, ref_end_upper = ref_end_interval

                if (
                    ref_start_lower <= seg_start <= ref_start_upper 
                    and ref_end_lower <= seg_end <= ref_end_upper
                ):
                    n_tokens_hit += 1
                    ref_intervals.pop(i_gt_word)  # can't re-use token
                    if strict: break

    return n_tokens_seg, n_tokens_ref, n_tokens_hit