from typing import Iterable, Tuple

import numpy as np

from utils.utils import *

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


def purity(C, calc_iper=False):
    N = np.sum(C)
    purity_ = np.sum(
        [np.max(C[:,k]) for k in range(C.shape[1])]
        ) / N if N > 0 else 0.0
    cluster_purity = np.sum(
        [np.max(C[k,:]) for k in range(C.shape[0])]
        ) / N if N > 0 else 0.0

    if not calc_iper:
        return purity_, cluster_purity, None
    
    disc_type_iper_val = np.sum(
        [np.max(C[k,:]) for k in range(C.shape[0]) 
        if np.sum(C[k,:]) > 1]
        ) / np.sum(
        [np.sum(C[k,:]) for k in range(C.shape[0]) 
        if np.sum(C[k,:]) > 1]
        )

    return purity_, cluster_purity, disc_type_iper_val


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

    # word/syl-normalized mutual information
    wnmi = mi / Hc

    # cluster-normalized mutual information
    cnmi = mi / Hk

    return mi, nmi, wnmi, cnmi


def f1_score(precision: float, recall: float) -> float:
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


def eulc_dist(precision: float, recall: float) -> float:
    return 1-np.sqrt(precision**2 + recall**2)