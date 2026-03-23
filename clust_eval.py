"""
Funtions used to evaluate a lexicon learning algorithm. Called as main script, it evaluates the input segmentation file.

Author: Benjamin van Niekerk, Simon Malan
Contact: benjamin.l.van.niekerk@gmail.com, 24227013@sun.ac.za
Date: October 2025
"""

import argparse
from pathlib import Path
import itertools
from tqdm import tqdm

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

USED_MODAL_TYPES = []

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
    num_elements = len(group)
    if num_elements == 1: # pop singleton classes for reverse metric
        return [(np.nan, num_elements)]
    
    edit_dist = [editdistance.eval(p, q)/max(len(p), len(q))
                 if max(len(p), len(q)) > 0 else 1
                 for p, q in itertools.combinations(group, 2)]
    
    if norm: return [(statistics.mean(edit_dist), num_elements)] # Average per cluster
    return [(edit_dist_, num_elements) for edit_dist_ in edit_dist]


def reverse_ned(gt_word_dict, norm: bool = False) -> float:
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
    distances = [
        (d, l)
        for clusters in gt_word_dict
        for d, l in distance([tuple(item[1]) for item in gt_word_dict[clusters]], norm=norm)
    ]

    if norm: # weighted mean: sum over classes
        weights = [l for _, l in distances if l != 1] # pop singleton classes
        distances = [d for d, l in distances if l != 1] # pop singleton classe
        norm_per_num_pairwise = np.average(distances, weights=weights)
    else:
        distances = [d for d, l in distances if l != 1] # pop singleton classes

    if norm:
        return np.nanmean(distances) if len(distances) > 0 else 0.0, norm_per_num_pairwise 
    else:
        return np.nanmean(distances) if len(distances) > 0 else 0.0, _


def ned(discovered: Iterable[Tuple[Fragment, Transcription, int]], norm: bool = False) -> float:
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
    distances = [
        (d, l)
        for _, group in itertools.groupby(discovered, key=lambda x: x[2])
        for d, l in distance([c[1].tokens for c in group], norm=norm)
    ]

    if norm: # weighted mean: sum over clusters
        weights = [l for _, l in distances]
        distances_ = [d if l > 1 else 0.0 for d, l in distances]
        norm_per_num_pairwise = np.average(distances_, weights=weights)

    distances = [d for d, _ in distances]
    if norm:
        return np.nanmean(distances) if len(distances) > 0 else 0.0, norm_per_num_pairwise # Per-cluster Norm NED, Per-cluster # elements Norm NED
    else:
        return np.nanmean(distances) if len(distances) > 0 else 0.0, _ # Original NED, _


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
        current_distances = [editdistance.eval(i, candidate_modal_class) / longest_modes_len for i in group]
        current_total = np.sum(current_distances)
        
        if current_total < min_total_dist:
            min_total_dist = current_total
            best_distances = current_distances
    
    return best_distances


def reverse_per(gt_word_dict, reverse: bool = True) -> float:
    """
    Not normalized (average over all comparisons)
    Pop singleton classes
    """
    distances = [
        d
        for clusters in gt_word_dict
        for d in phone_edit_distance([tuple(p[1]) for p in gt_word_dict[clusters]], reverse=reverse)
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


def purity(C):
    N = np.sum(C)
    purity_ = np.sum([np.max(C[:,k]) for k in range(C.shape[1])]) / N if N > 0 else 0.0
    cluster_purity = np.sum([np.max(C[k,:]) for k in range(C.shape[0])]) / N if N > 0 else 0.0

    return purity_, cluster_purity


def mutial_information(C):
    N = np.sum(C)
    
    # joint probabilities
    p_ck = C / N + 1e-10

    # marginal probabilities
    p_c = np.sum(p_ck, axis=1)
    p_k = np.sum(p_ck, axis=0)

    # mutual information
    mi = (p_ck * np.log(p_ck / p_c[:, None] / p_k[None, :])).sum()

    # normalized mutual information
    Hc = -(p_c * np.log(p_c)).sum()
    Hk = -(p_k * np.log(p_k)).sum()
    nmi = mi / np.average([Hc, Hk])

    # word-normalized mutual information
    wnmi = mi / Hc

    # cluster-normalized mutual information
    cnmi = mi / Hk

    return mi, nmi, wnmi, cnmi

def homogeneity_completeness(labels, C, measure):
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
    gold: Iterable[Transcription],
    disc: Iterable[Tuple[Fragment, Transcription]],
):
    covered = {
        (fragment.speaker, interval.begin, interval.end, interval.data)
        for fragment, transcription, _ in disc
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
    disc_types = {transcription.tokens for _, transcription, _ in disc}
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


def treeify(grid: TextGrid, tier=1, sub = "\d") -> IntervalTree:
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
        # if all(interval.data.lower() not in ["sil","spn","sp","","<unk>"] for interval in intervals)
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

    # if transcription == [] and fragment.interval.end - fragment.interval.begin != 0.02: print("No matching intervals for fragment:", fragment, sorted(tree.overlap(fragment.interval), key=lambda x: x.begin))

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

    # Segment-type (words or syllables)
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
    for file in files:
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
    disc_info = [
        (f := Fragment(speaker, interval), transcribe(f, phone_trees[f.speaker]), clust)
        for speaker, interval, clust in fragments
    ]

    # Remove silence, speaker noise, and empty transcriptions for discovered segments
    count_words_NS = 0
    cluster_set = set()
    pop_list = []
    for i_trans, (_, transcription, cluster) in enumerate(disc_info):
        cluster_set.add(cluster)
        non_silence = False
        for i_int, interval in enumerate(transcription.intervals):
            if interval.data.lower() not in ["sil", "spn", "sp", ""]:
                non_silence = True
        if non_silence:
            count_words_NS += 1
        else:
            pop_list.append(i_trans) # Remove if all phone tokens are silence or speaker noise or empty
    print("Number of clusters:", len(cluster_set))    

    if len(pop_list) > 0:
        print("Silences/unmapped fragments found!", len(pop_list))
        indices_to_drop = set(pop_list)
        disc_info = [v for i, v in enumerate(disc_info) if i not in indices_to_drop]
    print("Number of discovered units:", count_words_NS)

    # For reverse NED and PER
    gt_word_dict = {}
    total_duration = 0.0
    disc_info = sorted(disc_info, key=lambda x: (x[0].speaker, x[0].interval[0]))   
    for word_tree, group in itertools.groupby(tqdm(disc_info), key=lambda x: x[0].speaker):
        
        if group_by_words:
            gt_phones = words(grids[word_tree], phone_trees[word_tree], tier=0) # GT phone-level transcription of all words in the utterance
        else:
            gt_phones = words(grids[word_tree], phone_trees[word_tree], tier=2) # GT phone-level transcription of all syllables in the utterance
        
        disc_transcription = []
        for _, disc_phone_seq, cluster in group:
            interval_set = frozenset(disc_phone_seq.intervals)
            disc_transcription.append((disc_phone_seq, cluster, interval_set))
        disc_len = len(disc_transcription)
        disc_idx = 0

        for gt_phone in gt_phones: # per GT word/syl
            clusters = []
            gt_seq_begin, gt_seq_end = gt_phone.intervals[0].begin, gt_phone.intervals[-1].end
            
            for gt_phone_interval in gt_phone.intervals:
                total_duration += gt_phone_interval.end - gt_phone_interval.begin

                temp_idx = disc_idx
                while temp_idx < disc_len:
                    disc_phone_seq, cluster, interval_set = disc_transcription[temp_idx]
                    disc_start, disc_end = disc_phone_seq.intervals[0].begin, disc_phone_seq.intervals[-1].end

                    if (disc_start < gt_seq_end) and \
                    (disc_end > gt_seq_begin) and \
                    (gt_phone_interval in interval_set):
                        if not clusters or clusters[-1] != cluster:
                            clusters.append(cluster)
                    
                    if disc_end <= gt_phone_interval.end and temp_idx == disc_idx: # skip this disc interval in the next loop
                        disc_idx += 1
                    elif disc_start >= gt_phone_interval.end: # skip to the next phone in the gt sequence
                        break
                    temp_idx += 1

            if clusters:
                word_dict_key = " ".join(list(gt_phone.tokens)).strip()
                if word_dict_key not in gt_word_dict:
                    gt_word_dict[word_dict_key] = []
                gt_word_dict[word_dict_key].append((word_tree, clusters, gt_phone.intervals))

    print("Number of ground-truth phonetic realizations of words:", len(gt_word_dict))
    print("Number of singleton ground-truth phonetic realizations of words:", sum(1 for v in gt_word_dict.values() if isinstance(v, list) and len(v) == 1))
    print("Number of discovered segments mapped to ground truth phonetic realizations of words:", sum(len(v) for v in gt_word_dict.values()), "\n")

    word_transcriptions, disc_tokens, disc_clusters = [], [], []
    for frag, transc, clust in disc_info:
        word_transcriptions.append(
            transcribe(frag, word_trees[frag.speaker], max_overlap=True)
        )
        disc_tokens.append(
            " ".join(transc.tokens)
        )
        disc_clusters.append(clust)

    all_words = [" ".join(list(item.tokens)) for item in word_transcriptions]

    if group_by_words:
        gold_words = {speaker: words(grids[speaker], phone_trees[speaker], tier=0) for speaker in grids.keys()}
    else:
        gold_words = {speaker: words(grids[speaker], phone_trees[speaker], tier=2) for speaker in grids.keys()}

    gold_transcriptions = [word for words in gold_words.values() for word in words] # (tuple of phones contained within gold word boundaries)

    # Phonemic-sequence-based metrics
    print("~~~ Phonemic-sequence-based metrics ~~~")

    ned_value, _ = ned(disc_info)
    ned_value_norm, ned_value_weighted_norm = ned(disc_info, norm=True)
    reverse_ned_value, _ = reverse_ned(gt_word_dict)
    print("NED, Reverse NED, F1 NED:", ned_value, reverse_ned_value, f1_score(ned_value, reverse_ned_value))
    print("NED-Acc, Reverse NED-Acc, F1 NED-Acc:", 1-ned_value, 1-reverse_ned_value, f1_score(1-ned_value, 1-reverse_ned_value))
    reverse_ned_value_norm, reverse_ned_value_weighted_norm = reverse_ned(gt_word_dict, norm=True)
    print("Normalized: NED, Reverse NED, F1 NED:", ned_value_norm, reverse_ned_value_norm, f1_score(ned_value_norm, reverse_ned_value_norm))
    print("Normalized # elem: NED, Reverse NED, F1 NED:", 
          ned_value_weighted_norm, reverse_ned_value_weighted_norm, f1_score(ned_value_weighted_norm, reverse_ned_value_weighted_norm)
    )
    
    per_value = per(disc_info)
    reverse_per_value = reverse_per(gt_word_dict)
    print("\nPER, Reverse PER, F1 PER:", per_value, reverse_per_value, f1_score(per_value, reverse_per_value))
    print("PAcc, Reverse PAcc, F1 PAcc:", 1-per_value, 1-reverse_per_value, f1_score(1-per_value, 1-reverse_per_value))

    print("\nCoverage", coverage(gold_transcriptions, disc_info))
    print("Types", types(gold_transcriptions, disc_info))
    print("Bitrate", bitrate(disc_clusters, count_words_NS, total_duration))

    # Contingency matrix
    C_words = contingency_matrix(all_words, disc_clusters)
    C_phones = contingency_matrix(disc_tokens, disc_clusters)

    # Word-level metrics
    print("\n~~~ Word-level metrics ~~~")
    word_level_purity, word_level_cluster_purity = purity(C_words)
    print("(Per cluster) Purity, Reverse (per word) Purity, F1 Purity", (
        word_level_purity, word_level_cluster_purity, f1_score(word_level_purity, word_level_cluster_purity)
    ))
    print("Homogeneity, Completeness, V-measure", metrics.homogeneity_completeness_v_measure(all_words, disc_clusters))
    word_level_mi, word_level_nmi, word_level_wnmi, word_level_cnmi = mutial_information(C_words)
    print("Mutual Information, Normalized MI, Word-normalized MI, Cluster-normalized MI", (
        word_level_mi, word_level_nmi, word_level_wnmi, word_level_cnmi
    ))

    # Phone-level realization metrics
    print("\n~~~ Phone-level word realization metrics ~~~")
    phone_level_purity, phone_level_cluster_purity = purity(C_phones)
    print("(Per cluster) Purity, Reverse (per phone-level word realization) Purity, F1 Purity", (
        phone_level_purity, phone_level_cluster_purity, f1_score(phone_level_purity, phone_level_cluster_purity)
    ))
    print("Homogeneity, Completeness, V-measure", metrics.homogeneity_completeness_v_measure(disc_tokens, disc_clusters))
    phone_level_mi, phone_level_nmi, phone_level_wnmi, phone_level_cnmi = mutial_information(C_phones)
    print("Mutual Information, Normalized MI, Phone-realization-word-normalized MI, Cluster-normalized MI", (
        phone_level_mi, phone_level_nmi, phone_level_wnmi, phone_level_cnmi
    ))

    # Cluster sizes and distribution
    print("\n~~~ Cluster size statistics ~~~")
    cluster_sizes = np.sum(C_words, axis=0)
    print("Mean, Median, Std, Max, Min:", 
        np.mean(cluster_sizes), np.median(cluster_sizes), np.std(cluster_sizes), np.max(cluster_sizes), np.min(cluster_sizes)
    )
    print("Number of singletons:", len(np.where(cluster_sizes == 1)[0]))