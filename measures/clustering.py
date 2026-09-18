from typing import Tuple

import numpy as np

from utils.utils import *


def coverage(
    num_gold: int,
    disc: List[Tuple[Fragment, Transcription, int]],
) -> float:
    """Calculates the coverage of the discovered units given the gold 
    transcriptions. Coverage is the proportion of gold phones that are 
    discovered.

    Parameters
    ----------
    num_gold : int
        The number of gold phones in the data.
    disc : list of tuple of (Fragment, Transcription, int)
        Discovered fragments each containing a ``Fragment``, its 
        ``Transcription``, and its cluster ID.

    Returns
    -------
    coverage : float
        The coverage value.
    """

    covered = {
        (fragment.speaker, interval.begin, interval.end, interval.data)
        for fragment, transcription, _ in disc
        for interval in transcription.intervals
        if interval.data.lower() not in ["sil","spn","sp",""]
    }
    return len(covered) / num_gold


def types(
    gold: List[Transcription],
    disc: List[Tuple[Fragment, Transcription, int]],
) -> Tuple[float, float, float]:
    """Calculates the type scores for a discovered segmentation.

    Parameters
    ----------
    gold : list of Transcription
        The transcription of each gold token.
    disc : list of tuple of (Fragment, Transcription, int)
        Discovered fragments each containing a ``Fragment``, its 
        ``Transcription``, and its cluster ID.

    Returns
    -------
    precision : float
        The number of discovered types that are also gold types.
    recall : float
        The number of gold types that are discovered.
    fscore : float
        F1 type-score value.
    """

    gold_types = {transcription.tokens for transcription in gold}
    disc_types = {transcription.tokens for _, transcription, _ in disc}
    intersection = gold_types & disc_types
    precision = len(intersection) / len(disc_types)
    recall = len(intersection) / len(gold_types)
    fscore = f1_score(precision, recall)
    return precision, recall, fscore


def bitrate(clusters: List[int], tot_counts: int, total_dur: float) -> float:
    """Calculates the bitrate of a lexicon. Uses the entropic formulation: the
    lower bound on the average number of bits required to encode the tokenized 
    speech.

    Parameters
    ----------
    clusters : list of int
        The clusters that the discovered units are assigned to.
    tot_counts : int
        The number of discovered units.
    total_dur : float
        The duration of the data in seconds.

    Returns
    -------
    bitrate : float
        The bitrate value.
    """

    clus_counts = np.unique(clusters, return_counts=True)[1]
    clus_probs = clus_counts / tot_counts
    clus_probs = clus_probs[clus_probs != 0]
    return -tot_counts*np.sum((clus_probs * np.log2(clus_probs))/total_dur)


def purity(C, calc_iper: bool=False) -> Tuple[float, float, Union[float|None]]:
    """Calculates the purity and inverse purity of a lexicon (class and cluster
    mapping). Also calculates the iPER metric if discovered classes are used.
    
    Parameters
    ----------
    C : {array-like, sparse}, shape=[n_classes, n_clusters]
        The contingency matrix with rows containing classes and columns
        containing clusters.
    calc_iper : bool
        If the gold classes are set to the discovered classes, this flag is set
        to ``True`` and the iPER value is calculated here to save compute.

    Returns
    -------
    purity_ : float
        Purity value. Average proportion of each cluster that matches the modal
        class in the cluster. 
    inverse_purity : float
        Inverse purity value. Average proportion of each class that matches the 
        modal cluster assigned to the class.
    disc_type_iper_val : float
        Inverse phone error rate for discovered classes. Singleton discovered
        classes are ignored.
    """

    N = np.sum(C)
    purity_ = np.sum(
        [np.max(C[:,k]) for k in range(C.shape[1])]
        ) / N if N > 0 else 0.0
    inverse_purity = np.sum(
        [np.max(C[k,:]) for k in range(C.shape[0])]
        ) / N if N > 0 else 0.0

    if not calc_iper:
        return purity_, inverse_purity, None
    
    disc_type_iper_val = np.sum(
        [np.max(C[k,:]) for k in range(C.shape[0]) 
        if np.sum(C[k,:]) > 1]
        ) / np.sum(
        [np.sum(C[k,:]) for k in range(C.shape[0]) 
        if np.sum(C[k,:]) > 1]
        )

    return purity_, inverse_purity, disc_type_iper_val


def mutial_information(C) -> Tuple[float, float, float, float]:
    """Calculates the mutual information of the lexicon (class and cluster 
    mapping) as well as three normalized versions.
    
    Parameters
    ----------
    C : {array-like, sparse}, shape=[n_classes, n_clusters]
        The contingency matrix with rows containing classes and columns
        containing clusters.
    
    Returns
    -------
    mi : float
        Mutial information value.
    nmi : float
        Normalized mutial information value.
    cnmi : float
        Class normalized mutial information value.
    knmi : float
        Cluster normalized mutial information value.
    """

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

    # class-normalized mutual information
    cnmi = mi / Hc

    # cluster-normalized mutual information
    knmi = mi / Hk

    return mi, nmi, cnmi, knmi


def f1_score(precision: float, recall: float) -> float:
    """Calculates the F1-score given ``precision`` and ``recall``.

    Parameters
    ----------
    precision : float
        Precision value.
    recall : float
        Recall value.

    Returns
    -------
    f1 : float
        F1-score value
    """

    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


def eucl_dist(precision: float, recall: float) -> float:
    """Calculates the Euclidean distance of ``precision`` and ``recall`` to the
    point ``(1,1)``.

    Parameters
    ----------
    precision : float
        Precision value.
    recall : float
        Recall value.

    Returns
    -------
    eucl : float
        Euclidean distance value.
    """

    return 1-np.sqrt(precision**2 + recall**2)