"""
Funtions used to evaluate a segmentation algorithm. 
Called as main script, it evaluates the input segmentation file.

Author: Herman Kamper, Simon Malan
Contact: kamperh@gmail.com, 24227013@sun.ac.za
Date: March 2024
"""

import argparse
from pathlib import Path

from measures.boundaries import *
from utils.data_reader import Reader


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
        type=Union[int|float],
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
    if args.alignment_type == "syllables" and args.gold_format == ".txt":
        print("Syllables classes cannot be used with ZeroSpeech alignments, " \
        "reverting to word classes.")
        args.alignment_type = "words"

    rdr = Reader(args.disc_root, args.gold_root, args.disc_format, args.gold_format)
    seg_list, ref_list = rdr.disc_gold_to_pairs(
        args.alignment_type, 
        args.ms_per_frame, 
        args.tolerance, 
        frames, 
        args.split_utterances
    )

    # -------------- Calculate boundary evaluation metrics --------------

    n_seg, n_ref, n_hit = eval_boundaries(
        seg_list, 
        ref_list, 
        tolerance=args.tolerance, 
        strict=args.strict
    )
    precision, recall, f1_score = get_p_r_f1(n_seg, n_ref, n_hit)

    os = get_os(n_seg, n_ref)
    rvalue = get_rvalue(os, recall)

    n_token_seg, n_token_ref, n_token_hit = eval_token_boundaries(
        seg_list, ref_list, tolerance=args.tolerance
    )
    token_p, token_r, token_f1 = get_p_r_f1(n_token_seg, n_token_ref, n_token_hit)

    print(f"Precision: {precision}, Recall: {recall}, F1: {f1_score}")
    print(f"Over-segmentation: {os}, R-value: {rvalue}")
    print(f"Token Precision: {token_p}, Token Recall: {token_r}, Token F1: {token_f1}")