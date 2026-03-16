"""
Funtions used to evaluate a lexicon learning algorithm. Called as main script, it evaluates the input segmentation file.

Author: Benjamin van Niekerk, Simon Malan
Contact: benjamin.l.van.niekerk@gmail.com, 24227013@sun.ac.za
Date: October 2025
"""

import argparse
from pathlib import Path
import itertools

import dataclasses
from typing import Iterable, List, Tuple
from intervaltree import IntervalTree, Interval
from textgrid import TextGrid, IntervalTier

import re
import numpy as np
import statistics
import editdistance
from sklearn import metrics
from sklearn.metrics.cluster import contingency_matrix

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


def distance(group: Iterable[Tuple[str]], norm: bool = False) -> float:
    if len(group) == 1: return [np.nan]
    edit_dist = [editdistance.eval(p, q)/max(len(p), len(q)) 
                 if max(len(p), len(q)) > 0 else 1 
                 for p, q in itertools.combinations(group, 2)]
    if norm: return [statistics.mean(edit_dist)]
    return edit_dist


def reverse_ned(gt_word_dict: dict, norm: bool = False) -> float:
    distances = [
        d
        for clusters in gt_word_dict
        for d in distance([tuple(item[1]) for item in gt_word_dict[clusters]], norm=norm)
    ]
    return np.nanmean(distances)


def ned(discovered: Iterable[Tuple[Fragment, int, Transcription]], norm: bool = False) -> float:
    discovered = sorted(discovered, key=lambda x: x[1])
    distances = [
        d
        for _, group in itertools.groupby(discovered, key=lambda x: x[1])
        for d in distance([c[2].tokens for c in group], norm=norm)
    ]
    return np.nanmean(distances) if len(distances) > 0 else 0.0


def phone_edit_distance(group: Iterable[Tuple[str]], norm: bool = False) -> Iterable[float]:
    if len(group) == 1: return [np.nan]
    modal_g = statistics.multimode(group)
    modal_g = modal_g[0] if len(modal_g) == 1 else max(modal_g, key=len)
    length = len(modal_g)
    assert length > 0
    if norm: return [statistics.mean([editdistance.eval(i, modal_g) / length for i in group])]
    return [editdistance.eval(i, modal_g) / length for i in group]


def reverse_per(gt_word_dict: dict, norm: bool = False) -> float:
    distances = [
        d
        for clusters in gt_word_dict
        for d in phone_edit_distance([tuple(p[1]) for p in gt_word_dict[clusters]], norm=norm)
    ]
    return np.nanmean(distances)


def per(discovered: Iterable[Tuple[int, Transcription]], norm: bool = False) -> float:
    discovered = sorted(discovered, key=lambda x: x[0])
    distances = [
        d
        for _, group in itertools.groupby(discovered, key=lambda x: x[0])
        for d in phone_edit_distance([c[1].tokens for c in group], norm=norm)
    ]
    return np.nanmean(distances) if len(distances) > 0 else 0.0


def purity(C):
    N = np.sum(C)
    purity_ = np.sum([np.max(C[:,k]) for k in range(C.shape[1])]) / N if N > 0 else 0.0
    cluster_purity = np.sum([np.max(C[k,:]) for k in range(C.shape[0])]) / N if N > 0 else 0.0

    return purity_, cluster_purity


def homogeneity_completeness(labels: Iterable[str], C: Iterable, measure: str) -> float:
    probs = np.unique(labels, return_counts=True)[1] / len(labels)
    entropy_C_K = np.sum(probs * np.log(probs))
    if entropy_C_K == 0:
        return 1.0
    
    N = np.sum(C)
    assert N == len(labels)
    H = 0.0
    if measure == "homogeneity":
        for c in range(C.shape[0]):
            for k in range(C.shape[1]):
                if C[c, k] > 0:
                    p_k = np.sum(C[:,k])/N # p(k)
                    p_cgk = C[c,k]/np.sum(C[:,k]) # p(c|k)
                    H -= p_k * p_cgk * np.log(p_cgk)
    elif measure == "completeness":
        for k in range(C.shape[1]):
            for c in range(C.shape[0]):
                if C[c, k] > 0:
                    p_c = np.sum(C[c,:])/N # p(c)
                    p_kgc = C[c,k]/np.sum(C[c,:]) # p(k|c)
                    H -= p_c * p_kgc * np.log(p_kgc) # H(k|c) weight by p(c)
    return 1 - (H / -entropy_C_K)

def f1_score(precision: float, recall: float) -> float:
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

def coverage(
    disc: Iterable[Tuple[Fragment, Transcription]],
    gold: Iterable[Transcription],
) -> float:
    covered = {
        (fragment.speaker, interval.begin, interval.end, interval.data)
        for fragment, transcription in disc
        for interval in transcription.intervals
        if interval.data.lower() not in ["sil","spn","sp",""]
    }
    total = [
        interval.data
        for transcription in gold
        for interval in transcription.intervals
        if interval.data.lower() not in ["sil","spn","sp",""]
    ]
    return len(covered) / len(total)


def types(
    gold: Iterable[Transcription],
    disc: Iterable[Transcription],
) -> Tuple[float, float, float]:
    gold_types = {transcription.tokens for transcription in gold}
    disc_types = {transcription.tokens for transcription in disc}
    intersection = gold_types & disc_types
    precision = len(intersection) / len(disc_types)
    recall = len(intersection) / len(gold_types)
    fscore = 2 * (precision * recall) / (precision + recall)
    return precision, recall, fscore


def tokens(
    gold: Iterable[Fragment],
    disc: Iterable[Fragment],
) -> Tuple[float, float, float]:
    gold_fragments = set(gold)
    disc_fragments = set(disc)
    intersection = gold_fragments & disc_fragments
    precision = len(intersection) / len(disc_fragments)
    recall = len(intersection) / len(gold_fragments)
    fscore = 2 * (precision * recall) / (precision + recall)
    return precision, recall, fscore


def bitrate(classes: Iterable[int], tot_counts: int, total_dur: float) -> float:
    clus_counts = np.unique(classes, return_counts=True)[1]
    clus_probs = clus_counts / tot_counts
    clus_probs = clus_probs[clus_probs != 0]
    return -tot_counts*np.sum((clus_probs * np.log2(clus_probs))/total_dur)


def check_boundary(gold: Interval, disc: Interval) -> bool:
    if gold.contains_interval(disc):
        return True

    gold_duration = round(gold.end - gold.begin, 2)
    overlap_duration = round(gold.overlap_size(disc), 2)
    overlap_percentage = overlap_duration / gold_duration
    duration_condition = gold_duration >= 0.06 and overlap_duration >= 0.03
    percentage_condition = gold_duration < 0.06 and overlap_percentage >= 0.5
    return duration_condition or percentage_condition


def treeify(grid: TextGrid, tier: int = 1, sub: str = "\d") -> IntervalTree:
    intervals = [
        (interval.minTime, interval.maxTime, re.sub(sub, "", interval.mark))
        for interval in grid.tiers[tier]
    ]
    return IntervalTree.from_tuples(intervals)


def words(grid: TextGrid, tree: IntervalTree, tier: int) -> List[Transcription]:
    overlaps = [
        tree.overlap(interval.minTime, interval.maxTime)
        for interval in grid.tiers[tier]
        if interval.mark != "<eps>"
    ]
    overlaps = [
        sorted(intervals, key=lambda x: x.begin)
        for intervals in overlaps
        if all(interval.data.lower() not in ["sil","spn","sp","","<unk>"] for interval in intervals)
    ]
    overlaps = [Transcription(intervals) for intervals in overlaps]
    return overlaps


def transcribe(fragment: Fragment, tree: IntervalTree, max_overlap: bool = False) -> Transcription:
    transcription = sorted(tree.overlap(fragment.interval), key=lambda x: x.begin)
    
    # Return the maximally overlapping ground-truth interval
    if max_overlap:
        if len(transcription) == 1: return Transcription(transcription)
        overlaps = [fragment.interval.overlap_size(interval) for interval in transcription]
        max_overlap = max(overlaps)
        # if multiple max overlaps, choose the one that is the largest proportion of its own length
        if overlaps.count(max_overlap) > 1:
            proportions = [
                overlap / (interval.end - interval.begin) if overlap == max_overlap else 0
                for overlap, interval in zip(overlaps, transcription)
            ]
            max_overlap = max(proportions)
            overlaps = proportions

        max_idx = overlaps.index(max_overlap)
        transcription = [transcription[max_idx]]
        return Transcription(transcription)
    
    transcription = [
        interval
        for interval in transcription
        if check_boundary(interval, fragment.interval)
    ]

    if transcription == []: print("No matching intervals for fragment:", fragment, sorted(tree.overlap(fragment.interval), key=lambda x: x.begin))

    return Transcription(transcription)


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
        "--group_by",
        metavar="--alignment-format",
        help="what level of transcription to use for reverse metrics.",
        default="words",
        type=str,
    )
    parser.add_argument(
        "--alignment_format",
        metavar="--alignment-format",
        help="extension of the alignment files.",
        default=".TextGrid",
        type=str,
    )
    args = parser.parse_args()

    group_by_words = True if args.group_by == "words" else False

    # Load discovered fragments
    files = args.disc_path.rglob("**/*" + ".list")
    fragments = []
    for file in files:
        with open(file, "r") as f:
            start_time = 0.0
            for line in f:
                if len(line.split(" ")) == 2: # end_time class
                    end_time, cluster = line.split(" ")
                    speaker = file.stem
                    if "None" in cluster or cluster is None: # skip if no cluster (removing silence scenarios)
                        start_time = float(end_time)
                        continue
                    fragments.append((speaker, Interval(float(start_time), float(end_time)), int(cluster)))
                    start_time = float(end_time)
    
    # Load gold alignments
    grids = {}
    files = args.gold_dir.rglob("**/*" + args.alignment_format)
    sub = "" if "mandarin" in str(args.gold_dir).lower() else "\d" # Add stress factors for Mandarin (same as ZRC)
    for file in files: # alignment files
        if args.alignment_format == ".TextGrid":
            grids[file.stem] = TextGrid.fromFile(file)
        elif args.alignment_format == ".txt": # ZRC format
            tier = 0
            with open(file, "r") as f:
                if file.stem not in grids: # Create a new TextGrid for each file
                    grids[file.stem] = TextGrid()
                if "phone" in str(file):
                    interval_tier = IntervalTier(name="phones")
                elif "word" in str(file):
                    interval_tier = IntervalTier(name="words")
                for line in f:
                    line = line.split()
                    interval_tier.add(float(line[0]), float(line[1]), line[2])
                grids[file.stem].append(interval_tier)
    
    # Ground-truth trees for phone-sequences and words
    phone_trees = {speaker: treeify(grid, tier=1, sub=sub) for speaker, grid in grids.items()}
    if group_by_words:
        word_trees = {speaker: treeify(grid, tier=0, sub=sub) for speaker, grid in grids.items()}
    else:
        word_trees = {speaker: treeify(grid, tier=2, sub=sub) for speaker, grid in grids.items()}

    # Discovered segments
    disc_fragments = [
        Fragment(speaker, interval) for speaker, interval, _ in fragments
    ]
    disc_transcriptions = [ # phone-sequence transcription of discovered segments
        transcribe(fragment, phone_trees[fragment.speaker]) for fragment in disc_fragments
    ]
    disc_clusters = [cluster for _, _, cluster in fragments]
    print("Number of clusters:", len(set(disc_clusters)))

    # Remove silence, speaker noise, and empty transcriptions
    count_words_NS = 0
    pop_list = []
    for i_trans, transcription in enumerate(disc_transcriptions):
        non_silence = False
        for i_int, interval in enumerate(transcription.intervals):
            if interval.data.lower() not in ["sil", "spn", "sp", ""]:
                non_silence = True
        if non_silence:
            count_words_NS += 1
        else:
            pop_list.append(i_trans) # Remove if all phone tokens are silence or speaker noise

    if len(pop_list) > 0:
        print("Silences/unmapped fragments found!", len(pop_list))
        for i in sorted(pop_list, reverse=True):
            disc_transcriptions.pop(i)
            disc_clusters.pop(i)
            disc_fragments.pop(i)
    print("Number of discovered units:", count_words_NS)

    word_transcriptions = [ # The word that mostly overlaps with discovered fragments
    transcribe(fragment, word_trees[fragment.speaker], max_overlap=True) for fragment in disc_fragments
    ]
    all_words = [" ".join(list(item.tokens)) for item in word_transcriptions]
    disc_tokens = [" ".join(transcription.tokens) for transcription in disc_transcriptions]

    if group_by_words:
        gold_words = {speaker: words(grids[speaker], phone_trees[speaker], tier=0) for speaker in grids.keys()}
    else:
        gold_words = {speaker: words(grids[speaker], phone_trees[speaker], tier=2) for speaker in grids.keys()}

    gold_transcriptions = [word for words in gold_words.values() for word in words] # (tuple of phones contained within gold word boundaries)

    # For reverse NED and PER
    gt_word_dict = {}
    total_duration = 0.0
    for word_tree in sorted(word_trees.keys()):
        if group_by_words:
            gt_phones = words(grids[word_tree], phone_trees[word_tree], tier=0) # GT phone-level transcription of all words in the utterance
        else:
            gt_phones = words(grids[word_tree], phone_trees[word_tree], tier=2) # GT phone-level transcription of all syllables in the utterance

        disc_transcription = [
            (disc_clusters[i_fragment], disc_transcriptions[i_fragment]) 
            for i_fragment, disc_fragment in enumerate(disc_fragments) if disc_fragment.speaker == word_tree
        ]
        for gt_phone in gt_phones: # per GT word
            clusters = []
            for gt_phone_interval in gt_phone.intervals:
                total_duration += gt_phone_interval.end - gt_phone_interval.begin
                for cluster, disc_phone in disc_transcription:
                    if (disc_phone.intervals[0].begin < gt_phone.intervals[-1].end) and \
                    (disc_phone.intervals[-1].end > gt_phone.intervals[0].begin) and \
                    (gt_phone_interval in disc_phone.intervals) and \
                    (clusters[-1] != cluster if len(clusters) > 0 else True):
                        clusters.append(cluster)
            word_dict_key = (" ".join(list(gt_phone.tokens))).strip()
            if word_dict_key not in gt_word_dict:
                gt_word_dict[word_dict_key] = []
            gt_word_dict[word_dict_key].append((word_tree, clusters, gt_phone.intervals))
    print("Number of ground-truth phonetic realizations of words:", len(gt_word_dict), "\n")

    # Phonemic-sequence-based metrics
    print("~~~ Phonemic-sequence-based metrics ~~~")

    # Evaluation metrics
    ned_value = ned(zip(disc_fragments, disc_clusters, disc_transcriptions))
    reverse_ned_value = reverse_ned(gt_word_dict)
    print("NED, Reverse NED, F1 NED:", ned_value, reverse_ned_value, f1_score(ned_value, reverse_ned_value))
    ned_value_norm = ned(zip(disc_fragments, disc_clusters, disc_transcriptions), norm=True)
    reverse_ned_value_norm = reverse_ned(gt_word_dict, norm=True)
    print("Normalized: NED, Reverse NED, F1 NED:", ned_value_norm, reverse_ned_value_norm, f1_score(ned_value_norm, reverse_ned_value_norm))
    print("Normalized: NAcc, Reverse NAcc, F1 NAcc:", 1-ned_value_norm, 1-reverse_ned_value_norm, f1_score(1-ned_value_norm, 1-reverse_ned_value_norm))

    per_value = per(zip(disc_clusters, disc_transcriptions))
    reverse_per_value = reverse_per(gt_word_dict)
    print("\nPER, Reverse PER, F1 PER:", per_value, reverse_per_value, f1_score(per_value, reverse_per_value))
    print("PAcc, Reverse PAcc, F1 PAcc:", 1-per_value, 1-reverse_per_value, f1_score(1-per_value, 1-reverse_per_value))
    per_value_norm = per(zip(disc_clusters, disc_transcriptions), norm=True)
    reverse_per_value_norm = reverse_per(gt_word_dict, norm=True)
    print("Normalized: PER, Reverse PER, F1 PER:", per_value_norm, reverse_per_value_norm, f1_score(per_value_norm, reverse_per_value_norm))
    print("Normalized: PAcc, Reverse PAcc, F1 PAcc:", 1-per_value_norm, 1-reverse_per_value_norm, f1_score(1-per_value_norm, 1-reverse_per_value_norm))

    print("\nCoverage", coverage(zip(disc_fragments, disc_transcriptions), gold_transcriptions))
    print("Types", types(gold_transcriptions, disc_transcriptions))
    print("Bitrate", bitrate(disc_clusters, count_words_NS, total_duration))

    # Contingency matrix
    C_words = contingency_matrix(all_words, disc_clusters)
    C_phones = contingency_matrix(disc_tokens, disc_clusters)

    # Word-level metrics
    print("\n~~~ Word-level metrics ~~~")
    word_level_purity, word_level_cluster_purity = purity(C_words)
    print("(Per cluster) Purity, Reverse (per word) Purity, F1 Purity", 
        word_level_purity, word_level_cluster_purity, f1_score(word_level_purity, word_level_cluster_purity)
    )
    h, c, v = metrics.homogeneity_completeness_v_measure(all_words, disc_clusters)
    print("Homogeneity, Completeness, V-measure", h, c, v)

    # Phone-level realization metrics
    print("\n~~~ Phone-level word realization metrics ~~~")
    phone_level_purity, phone_level_cluster_purity = purity(C_phones)
    print("(Per cluster) Purity, Reverse (per phone-level word realization) Purity, F1 Purity", 
        phone_level_purity, phone_level_cluster_purity, f1_score(phone_level_purity, phone_level_cluster_purity)
    )
    h, c, v = metrics.homogeneity_completeness_v_measure(disc_tokens, disc_clusters)
    print("Homogeneity, Completeness, V-measure", h, c, v)

    # Cluster sizes and distribution
    print("\n~~~ Cluster size statistics ~~~")
    cluster_sizes = np.sum(C_words, axis=0)
    print("Mean, Median, Std, Max, Min:", 
        np.mean(cluster_sizes), np.median(cluster_sizes), np.std(cluster_sizes), np.max(cluster_sizes), np.min(cluster_sizes)
    )

    # Own implementation of homogeneity and completeness, slower than sklearn
    # print("Word Homogeneity", homogeneity_completeness(all_words, C_words, "homogeneity"))
    # print("Word Completeness", homogeneity_completeness(disc_clusters, C_words, "completeness"))
    # print("Phone Homogeneity", homogeneity_completeness(disc_tokens, C_phones, "homogeneity"))
    # print("Phone Completeness", homogeneity_completeness(disc_clusters, C_phones, "completeness"))