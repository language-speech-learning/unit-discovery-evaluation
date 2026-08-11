"""
The lexicon evaluation toolkit from:
"Revisiting Lexicon Evaluation in Unsupervised Word Discovery".

Authors: Simon Malan, Benjamin van Niekerk, Danel Slabbert
Date: October 2025
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn import metrics
from collections import Counter
from sklearn.metrics.cluster import contingency_matrix

from utils.utils import *
from utils.data_reader import *
from utils.metric_schema import *
from measures.general import *
from measures.lexicon_specific import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "disc_root",
        metavar="disc-root",
        help="path to the root of the discovered fragments.",
        type=Path,
    )
    parser.add_argument(
        "gold_root",
        metavar="gold-root",
        help="path to the root of the alignments.",
        type=Path,
    )
    parser.add_argument(
        "--class_type",
        metavar="--class-type",
        help="what type to use as classes.",
        choices=["words", "syllables", "disc"],
        default="words",
        type=str,
    )
    parser.add_argument(
        "--disc_format",
        metavar="--disc-format",
        help="extension of the discovered fragments.",
        default=".list",
        type=str,
    )
    parser.add_argument(
        "--gold_format",
        metavar="--gold-format",
        help="extension of the alignment files.",
        default=".TextGrid",
        type=str,
    )
    args = parser.parse_args()

    gt_unit_tier = 0
    if args.class_type == "syllables":
        if args.gold_format == ".txt":
            print("Syllables classes cannot be used with ZeroSpeech alignments, " \
            "reverting to word classes.")
        else:
            gt_unit_tier = 2
    # Keep stress factors for Mandarin, remove for other languages (same as ZRC)
    sub = "" if "mandarin" in str(args.gold_root).lower() else r"\d" 

    # Read gold and discovered files
    rdr = Reader(args.disc_root, args.gold_root, args.disc_format, args.gold_format)
    grids, duration = rdr.gold_to_grids()
    fragments = rdr.disc_to_intervals()
    
    # Build gold interval structure
    phone_trees = {
        speaker: treeify(grid, tier=1, sub=sub) 
        for speaker, grid in grids.items()
    }

    # Build discovered segment structure
    disc_info = [
        (f := Fragment(speaker, interval), transcribe(f, phone_trees[f.speaker]), clust)
        for speaker, interval, clust in fragments
    ]

    # Remove silence, speaker noise, and empty transcriptions from discovered segments
    count_units_NS = 0
    pop_list = []
    for i_trans, (_, transcription, cluster) in enumerate(disc_info):
        non_silence = False
        for i_int, interval in enumerate(transcription.intervals):
            if interval.data.lower() not in ["sil", "spn", "sp", ""]:
                non_silence = True
        if non_silence:
            count_units_NS += 1
        else: # Remove if all phone tokens are silence or speaker noise or empty
            pop_list.append(i_trans)
       
    if len(pop_list) > 0:
        print(len(pop_list), "silences/unmapped fragments found!")
        indices_to_drop = set(pop_list)
        disc_info = [v for i, v in enumerate(disc_info) if i not in indices_to_drop]

    disc_info = sorted(disc_info, key=lambda x: (x[0].speaker, x[0].interval[0]))   
    
    # For inverse metrics over gold types
    if args.class_type != "disc":
        gt_word_dict = get_inverse_transcription(
            disc_info, 
            grids, 
            phone_trees, 
            gt_unit_tier
        )

        # For general clustering metrics using maximum overlap rule
        gt_unit_trees = {
            speaker: treeify(grid, tier=gt_unit_tier, sub=sub) 
            for speaker, grid in grids.items()
        }

    disc_tokens, disc_clusters = [], []
    for frag, transc, clus in disc_info:
        disc_clusters.append(clus)
        # Map discovered fragments to class using max overlap rule
        if args.class_type != "disc":
            disc_tokens.append(
                " ".join(
                        transcribe(
                        frag, 
                        gt_unit_trees[frag.speaker], 
                        max_overlap=True
                    ).tokens
                )
            )
        else: # For discovered types classes
            disc_tokens.append(" ".join(transc.tokens))

    C = contingency_matrix(disc_tokens, disc_clusters)

    # Gold types for type scores 
    # (if classes are discovered types, default to words as gold types)
    gt_phone_realizations_of_types, gt_tokens = [], []
    for speaker in grids.keys():
        phone_realization = group_phones_by_tier(
            grids[speaker], phone_trees[speaker], tier=gt_unit_tier
        )
        gt_phone_realizations_of_types.extend(phone_realization)

        # Gold tokens (realizations of gold types)
        for frag in phone_realization:
            interv = frag.intervals
            gt_tokens.append(
                Fragment(speaker, Interval(interv[0].begin, interv[-1].end))
            )
    
    # General clustering metrics
    calc_iper = False if args.class_type != "disc" else True
    purity_val, inverse_purity_val, inverse_per_val = purity(C, calc_iper)
    vmeasure_vals = metrics.homogeneity_completeness_v_measure(
        disc_tokens, disc_clusters
    )
    mi, nmi, gold_unit_nmi, cluster_nmi = mutial_information(C)
    type_prec, type_rec, typef1 = types(gt_phone_realizations_of_types, disc_info)
    tok_prec, tok_rec, tok1 = tokens(gt_tokens, [frag for frag, _, _ in disc_info])
                
    # Lexicon-specific metrics
    bitrate_val = bitrate(disc_clusters, len(disc_clusters), duration)
    ned_val, _, weighted_ned_val = ned(disc_info)
    if args.class_type != "disc":
        inverse_ned_val, _, inverse_weighted_ned_val = inverse_ned(gt_word_dict)
    else:
        inverse_ned_val, inverse_weighted_ned_val = disc_type_ined(disc_info)
    per_val = per(disc_info)
    if args.class_type != "disc":
        inverse_per_val = inverse_per(gt_word_dict)
    else:
        inverse_per_val = 1-inverse_per_val
    f1_nes = f1_score(1-ned_val, 1-inverse_ned_val)
    f1_wnes = f1_score(1-weighted_ned_val, 1-inverse_weighted_ned_val)
    d_pacc = eulc_dist(per_val, inverse_per_val)

    # Cluster sizes and distribution
    if args.class_type != "disc":
        num_classes = len(gt_word_dict)
        num_single_classes = sum(
            1 for v in gt_word_dict.values() 
            if isinstance(v, list) and len(v) == 1
        )
    else:
        disc_type_counts = Counter(disc_tokens)
        num_classes = len(disc_type_counts)
        assert len(set(disc_tokens)) == num_classes # TODO remove
        num_single_classes = sum(
            1 for c in disc_type_counts.values()
            if c == 1
        )

    cluster_sizes = np.sum(C, axis=0)
    num_clust = C.shape[1]
    clust_mean = np.mean(cluster_sizes)
    clust_std = np.std(cluster_sizes)
    clust_med = int(np.median(cluster_sizes))
    clust_max = np.max(cluster_sizes)
    clust_min = np.min(cluster_sizes)
    clust_single = len(np.where(cluster_sizes == 1)[0])

    results = EvaluationResults(
        num_disc_units=count_units_NS,
        classes=clss(args.class_type, num_classes, num_single_classes),
        clusters=clust(num_clust, clust_single, clust_mean, clust_std, clust_med, clust_max, clust_min),
        coverage=coverage(gt_phone_realizations_of_types, disc_info),
        bitrate=bitrate_val,
        original_nes=fwd_inv_f1(1-ned_val, 1-inverse_ned_val, f1_nes),
        weighted_nes=fwd_inv_f1(1-weighted_ned_val, 1-inverse_weighted_ned_val, f1_wnes),
        pacc=fwd_inv_d(1-per_val, 1-inverse_per_val, d_pacc),
        purity=fwd_inv_f1(purity_val, inverse_purity_val, f1_score(purity_val, inverse_purity_val)),
        types=p_r_f1(type_prec, type_rec, typef1),
        tokens=p_r_f1(tok_prec, tok_rec, tok1),
        v_measure=v_m(vmeasure_vals[0], vmeasure_vals[1], vmeasure_vals[2]),
        mutual_info=mut_inf(mi, nmi, gold_unit_nmi, cluster_nmi),
    )

    # Print some metrics
    print_results(args.disc_root.stem, results)

    # Save all results to json
    json_name = f"scores/lexicon_scores_{args.disc_root.stem}"
    save_json(args.disc_root.stem, json_name, results)