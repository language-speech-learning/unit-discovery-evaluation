"""
Funtions used to evaluate a segmentation algorithm. Called as main script, it evaluates the input segmentation file.

Author: Herman Kamper, Simon Malan
Contact: kamperh@gmail.com, 24227013@sun.ac.za
Date: March 2024
"""

import numpy as np
import argparse
from pathlib import Path
import textgrids
from tqdm import tqdm
from typing import List, Tuple, Union
import itertools
    
def get_p_r_f1(
    n_seg: int, n_ref: int, n_hit: int
) -> Tuple[float, float, float]:
    """
    Calculate boundary precision, recall, F-score.

    Parameters
    ----------
    n_seg : int
        The number of segmentation boundaries.
    n_ref : int
        The number of reference boundaries.
    n_hit : int
        The number of seg-ref hits.

    Return
    ------
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

def get_os(
    n_seg: int, n_ref: int
) -> float:
    """
    Calculates over-segmentation: how many fewer/more boundaries are proposed compared to the ground-truth.

    Parameters
    ----------
    n_seg : int
        The number of segmentation boundaries.
    n_ref : int
        The number of reference boundaries.

    Return
    ------
    output : float
        Over-segmentation
    """

    if n_ref == 0:
        return -np.inf
    else:
        return n_seg/n_ref - 1
    
def get_rvalue(
    os: float, recall: float
) -> Tuple[float, float]:
    """
    Calculates the R-value: how close the segmentation is to an ideal point of operation (100% Recall with 0% OS).

    Parameters
    ----------
    os : float
        How many fewer/more boundaries are proposed compared to the ground-truth.
    recall : float
        How often the reference boundaries are contained in the segmentation.

    Return
    ------
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
    """
    Count number of seg-ref hits.

    Parameters
    ----------
    seg : list of list of shape (n_utterances, )
        Segmentation hypothesis to evaluate.
    ref : list of list of shape (n_utterances, )
        Ground-truth segmentation used as reference.
    tolerance : numerical
        The number of frames or seconds within which a seg boundary can hit a ref boundary.
        If `int`, interpreted as number of frames; if `float`, interpreted as number of seconds.
    strict : bool, default=True
        If `True`, a reference boundary can only be hit once. If `False`, a reference boundary can be hit multiple times.
    n_seg : int, default=0
        The number of segmentation to start from.
    n_ref : int, default=0
        The number of reference boundaries to start from.
    n_hit : int, default=0
        The number of hits to start from.

    Return
    ------
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
    """
    Count number of token (onset-offset) seg-ref hits.

    Parameters
    ----------
    seg : list of list of shape (n_utterances, )
        Segmentation hypothesis to evaluate.
    ref : list of list of shape (n_utterances, )
        Ground-truth segmentation used as reference.
    tolerance : numerical
        The number of frames or seconds within which a seg boundary can hit a ref boundary.
        If `int`, interpreted as number of frames; if `float`, interpreted as number of seconds.
    strict : bool, default=True
        If `True`, a reference boundary can only be hit once. If `False`, a reference boundary can be hit multiple times.
    n_tokens_seg : int, default=0
        The number of segmentation to start from.
    n_tokens_ref : int, default=0
        The number of reference boundaries to start from.
    n_tokens_hit : int, default=0
        The number of hits to start from.

    Return
    ------
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

                if (ref_start_lower <= seg_start <= ref_start_upper and ref_end_lower <= seg_end <= ref_end_upper):
                    n_tokens_hit += 1
                    ref_intervals.pop(i_gt_word)  # can't re-use token
                    if strict: break

    return n_tokens_seg, n_tokens_ref, n_tokens_hit

def split_utterance(
    seg: List[float], ref: textgrids.Interval, tolerance: Union[int, float]
) -> Tuple[List[List[Union[int, float]]], List[List[Union[int, float]]]]:
    """
    Split segmentation and reference into utterances based on silences in ground-truth reference intervals.
    Output format: no utterance onset boundary, intermediate boundaries, utterance offset boundary.

    Parameters
    ----------
    seg : list of float
        The segmentation boundaries.
    ref : textgrids.Interval
        The reference word intervals.
    tolerance : numerical
        The number of frames or seconds within which a seg boundary can hit a ref boundary.
        If `int`, interpreted as number of frames; if `float`, interpreted as number of seconds.
    
    Return
    ------
    output : (list of list of float, list of list of float)
        The segmentation and reference boundaries split into utterances.
    """
    
    ref_out = [
        list(ref_utt)
        for k, ref_utt in itertools.groupby(
            ref, lambda x: x.text != ""
        )
        if k
    ]

    seg_out = []
    for ref_utt in ref_out:
        if ref_utt[0].xmin == 0.0: # If no silence at start of utterance
            ref_utt_onset = ref_utt[0].xmin - 1e-8
        else:
            ref_utt_onset = ref_utt[0].xmin + tolerance
        ref_utt_offset = ref_utt[-1].xmax - tolerance
        seg_out.append([
            s for s in seg if ref_utt_onset < s < ref_utt_offset
        ])
        seg_out[-1].append(ref_utt[-1].xmax)
        
    return seg_out, [[float(interval.xmax) for interval in ref_utt] for ref_utt in ref_out]

def get_frame_num(
    seconds: np.ndarray, ms_per_frame: int
) -> np.ndarray:
    """
    Convert seconds to frame number: rounds to nearest integer.

    Parameters
    ----------
    seconds : np.ndarray
        The number of seconds to convert to frames.
    ms_per_frame : int
        The number of milliseconds per speech feature frame.

    Return
    ------
    output : np.ndarray
        The frame number corresponding to the imput number of seconds.
    """
    
    return np.floor(np.round((seconds / ms_per_frame * 1000), 1) + 0.5).astype(np.int32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "disc_path",
        metavar="disc-path",
        help="path to the discovered fragments.",
        type=Path,
    )
    parser.add_argument(
        "gold_dir",
        metavar="gold-dir",
        help="path to the directory of alignments.",
        type=Path,
    )
    parser.add_argument(
        "--alignment_format",
        metavar="--alignment-format",
        help="extension of the alignment files.",
        default=".TextGrid",
        type=str,
    )
    parser.add_argument(
        "--alignment_type",
        metavar="--alignment-type",
        help="type of alignment tier to use.",
        default="words",
        choices=["words", "syllables", "phones"],
        type=str,
    )
    parser.add_argument(
        "--ms_per_frame",
        metavar="--ms-per-frame",
        help="number of ms in a frame for the encoding.",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--tolerance",
        help="the tolerance in number of frames if int, else tolerance in seconds.",
        default=1,
        type=float,
    )
    parser.add_argument(
        "--split_utterances",
        help="only evaluate between silences in the reference.",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--strict",
        help="optional variable to follow strict evaluation.",
        default=True,
        type=bool,
    )
    args = parser.parse_args()

    frames = True if args.tolerance % 1 == 0 else False

    files_seg = sorted(args.disc_path.rglob("**/*.list"))
    files_ref = sorted(args.gold_dir.rglob(f'**/*{args.alignment_format}'))

    assert len(files_seg) == len(files_ref)
    for sp, rp in zip(files_seg, files_ref):
        assert sp.stem == rp.stem

    seg_list, ref_list = [], []
    for file_seg, file_ref in zip(tqdm(files_seg),files_ref):
        with open(file_seg, "r") as f:
            lines = f.readlines()
            seg_utt = [
                float(line.strip().split(" ")[0]) 
                if len(line.strip().split(" ")) == 2 
                else float(line.strip()) for line in lines
            ]
            seg_utt = seg_utt[1:] if seg_utt[0] == 0.0 else seg_utt

        if args.alignment_format == '.TextGrid':
            tg = textgrids.TextGrid(file_ref)[args.alignment_type]
        elif args.alignment_format == '.txt':
            assert args.alignment_type == 'words'
            tg = textgrids.Tier()
            with open(file_ref, 'r') as f:
                for line in f:
                    line = line.split()
                    tg.append(textgrids.Interval(line[2], float(line[0]), float(line[1])))

        if frames:
            seg_utt = get_frame_num(
                np.array(seg_utt), ms_per_frame=args.ms_per_frame
            ).tolist()
            tg = [textgrids.Interval(
                text=interval.text,
                xmin=get_frame_num(interval.xmin, ms_per_frame=args.ms_per_frame).item(),
                xmax=get_frame_num(interval.xmax, ms_per_frame=args.ms_per_frame).item()
            ) for interval in tg]
        
        if args.split_utterances:
            seg_utt, ref_utt = split_utterance(
                seg_utt, tg, tolerance=args.tolerance
            )
            seg_list.extend(seg_utt)
            ref_list.extend(ref_utt)
        else:
            seg_list.append(seg_utt)
            ref_list.append([interval.xmax for interval in tg])

    # -------------- Calculate boundary evaluation metrics --------------

    n_seg, n_ref, n_hit = eval_boundaries(seg_list, ref_list, tolerance=args.tolerance, strict=args.strict)
    precision, recall, f1_score = get_p_r_f1(n_seg, n_ref, n_hit)

    os = get_os(n_seg, n_ref)
    rvalue = get_rvalue(os, recall)

    n_token_seg, n_token_ref, n_token_hit = eval_token_boundaries(seg_list, ref_list, tolerance=args.tolerance)
    token_p, token_r, token_f1 = get_p_r_f1(n_token_seg, n_token_ref, n_token_hit)

    print(f"Precision: {precision}, Recall: {recall}, F1: {f1_score}")
    print(f"Over-segmentation: {os}, R-value: {rvalue}")
    print(f"Token Precision: {token_p}, Token Recall: {token_r}, Token F1: {token_f1}")