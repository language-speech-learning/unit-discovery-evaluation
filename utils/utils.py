from typing import List, Tuple, Union

import re
import itertools
import numpy as np

import dataclasses
from textgrid import TextGrid
from intervaltree import IntervalTree, Interval

import json


@dataclasses.dataclass(frozen=True)
class Fragment:
    speaker: str
    interval: Interval


@dataclasses.dataclass(frozen=True)
class Transcription:
    intervals: List[Interval]

    @property
    def tokens(self) -> Tuple[str, ...]:
        return tuple(
            interval.data
            for interval in self.intervals
            if interval.data.lower() not in ["sil","spn","sp","<unk>",""]
        )

    @property
    def bounds(self) -> Interval:
        return Interval(self.intervals[0].begin, self.intervals[-1].end)


def check_boundary(gold: Interval, disc: Interval) -> bool:
    """Determines if a gold interval should be added to the discovered unit's
    transcription using the ZeroSpeech overlap rule: 50% or 30 ms of their 
    intervals should overlap depending on the gold phone's duration.

    Parameters
    ----------
    gold : Interval
    disc : Interval

    Returns
    -------
    condition : bool
        ``True`` if the ``gold`` and ``disc`` intervals satisfy the ZeroSpeech
        overlap condition, else ``False``.
    """

    # If the discovered interval is fully contained in a gold interval (no overlap)
    if gold.contains_interval(disc):
        return True

    gold_duration = round(gold.end - gold.begin, 3)
    overlap_duration = round(gold.overlap_size(disc), 3)
    overlap_percentage = overlap_duration / gold_duration
    duration_condition = gold_duration >= 0.060 and overlap_duration >= 0.030
    percentage_condition = gold_duration < 0.060 and overlap_percentage >= 0.50
    condition = duration_condition or percentage_condition
    return condition


def transcribe(
        fragment: Fragment, tree: IntervalTree, max_overlap: bool = False
    ) -> Transcription:
    """Transcribe a discovered ``fragment`` using a gold ``tree``. Use
    ZeroSpeech overlap rule if ``max_overlap`` is ``False``, otherwise use
    maximum overlap rule.
    
    Parameters
    ----------
    fragment : Fragment[speaker, inteval]
        A single discovered fragment.
    tree : IntervalTree
        Containing ``Interval[onset, offset, text]`` for each interval in the 
        corrosponding ground-truth utterance alignment.
    max_overlap : bool
        Substring to Regex remove from interval text.

    Returns
    -------
    transcription : Transcription
        Containing each ``Interval[onset, offset, text]`` from ``tree`` with
        which it overlaps.
    """

    # All gold intervals fully/partially overlapping with the disc fragment
    transcription = sorted(tree.overlap(fragment.interval), key=lambda x: x.begin)
    
    # Find the maximally overlapping ground-truth interval
    if max_overlap:
        if len(transcription) == 1:
            return Transcription(transcription)

        overlaps = [
            fragment.interval.overlap_size(interval) 
            for interval in transcription
        ]
        assert len(overlaps) > 0
        max_overlap = max(overlaps)
        # if multiple max overlaps, choose the one that is the largest proportion of its own length
        if overlaps.count(max_overlap) > 1:
            proportions = [
                overlap / interval.length()
                if overlap == max_overlap else 0
                for overlap, interval in zip(overlaps, transcription)
            ]
            max_overlap = max(proportions)
            overlaps = proportions

        max_idx = overlaps.index(max_overlap)
        transcription = [transcription[max_idx]]
        return Transcription(transcription)
    
    # Return the ZeroSpeech overlapping ground-truth intervals
    transcription = [
        interval
        for interval in transcription
        if check_boundary(interval, fragment.interval)
    ]

    transcription = Transcription(transcription)
    return transcription


def treeify(grid: TextGrid, tier, sub = r"\d") -> IntervalTree:
    """Build an ``InvervalTree`` from a ``TextGrid``.

    Parameters
    ----------
    grid : TextGrid
    tier : str
        ``TextGrid`` tier to extract from.
    sub : str
        Substring to Regex remove from interval text.

    Returns
    -------
    intv_tree : IntervalTree
        Containing ``Interval[onset, offset, text]`` for each gold unit of the
        specified ``tier``.
    """

    intervals = [
        (interval.minTime, interval.maxTime, re.sub(sub, "", interval.mark))
        for interval in grid.tiers[tier]
    ]
    intv_tree = IntervalTree.from_tuples(intervals)
    return intv_tree


def group_phones_by_tier(
        grid: TextGrid, 
        tree: IntervalTree, 
        tier: int
    ) -> List[Transcription]:
    """Group gold phone-level invervals by the onset and offsets of larger
    gold units specified by ``tier``.
    
    Parameters
    ----------
    grid : TextGrid
    tree : IntervalTree
        Containing ``Interval[onset, offset, text]`` for each gold phone 
        interval in an utterance.
    tier : str
        ``TextGrid`` tier to extract from.

    Returns
    -------
    overlaps : List[Transcription]
        Containing gold phone intervals ``Interval[onset, offset, text]`` 
        grouped by the gold unit specified by ``tier``.
    """

    overlaps = [
        tree.overlap(interval.minTime, interval.maxTime)
        for interval in grid.tiers[tier]
        if interval.mark != "<eps>"
    ]
    overlaps = [
        sorted(intervals, key=lambda x: x.begin)
        for intervals in overlaps
        # if all(interval.data.lower() not in ["sil","spn","sp","","<unk>"] for interval in intervals)
    ]
    overlaps = [Transcription(intervals) for intervals in overlaps]
    return overlaps


def get_inverse_transcription(
        disc_info, 
        grids, 
        phone_trees, 
        gt_unit_tier
    ) -> dict:
    """
    Parameters
    ----------
    disc_info : List[Tuple[Fragment, Transcription, int]]
        For each discovered fragment.
    grids : List[TextGrid]
        A ``TextGrid`` for each utterance.
    phone_trees : dict
        Containing and ``IntervalTree`` for each utterance.
    gt_unit_tier : str
        The ``TextGrid`` tier to use for the gold units.

    Returns
    -------
    gt_unit_dict : dict
        Containing gold phone intervals ``Interval[onset, offset, text]`` 
        grouped by the gold unit specified by ``tier``.
    """

    gt_unit_dict = {}
    total_duration = 0.0
    for utt, group in itertools.groupby(disc_info, key=lambda x: x[0].speaker):
        
        # Gold phone-level transcriptions grouped by gold words/syllables
        gt_phones = group_phones_by_tier(
            grids[utt], phone_trees[utt], tier=gt_unit_tier
        )
        
        disc_transcription = []
        for _, disc_phone_seq, cluster in group:
            interval_set = frozenset(disc_phone_seq.intervals)
            disc_transcription.append((disc_phone_seq, cluster, interval_set))
        disc_len = len(disc_transcription)
        disc_idx = 0

        # Per gold word/syllable
        for gt_phone in gt_phones:
            clusters = []
            prev_disc_idx_added = -1
            gt_seq_begin = gt_phone.intervals[0].begin
            gt_seq_end = gt_phone.intervals[-1].end
            
            # Per phone in current gold word/syllable
            for gt_phone_interval in gt_phone.intervals:
                total_duration += gt_phone_interval.end - gt_phone_interval.begin

                temp_idx = disc_idx
                while temp_idx < disc_len:
                    disc_phone_seq, cluster, interval_set = disc_transcription[temp_idx]
                    disc_start = disc_phone_seq.intervals[0].begin
                    disc_end = disc_phone_seq.intervals[-1].end

                    # Add a cluster to the inverse transcription
                    if (disc_start < gt_seq_end) and \
                    (disc_end > gt_seq_begin) and \
                    (gt_phone_interval in interval_set):
                        if prev_disc_idx_added != temp_idx:
                            clusters.append(cluster)
                        elif not clusters or clusters[-1] != cluster:
                            clusters.append(cluster)
                        prev_disc_idx_added = temp_idx
                    
                    if disc_end <= gt_phone_interval.end and temp_idx == disc_idx: 
                        disc_idx += 1 # Skip this disc interval in the next loop
                    elif disc_start >= gt_phone_interval.end: 
                        break # Skip to the next phone in the gold sequence
                    temp_idx += 1

            if clusters:
                unit_dict_key = " ".join(list(gt_phone.tokens)).strip()
                if unit_dict_key not in gt_unit_dict:
                    gt_unit_dict[unit_dict_key] = []
                gt_unit_dict[unit_dict_key].append(
                    (utt, clusters, gt_phone.intervals)
                )

    return gt_unit_dict