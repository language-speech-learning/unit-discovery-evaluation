from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple, Union, Sequence

import textgrids
import numpy as np
from intervaltree import Interval
from collections import defaultdict

from utils.utils import get_frame_num, split_utterance


class Reader:
    def __init__(
        self, 
        disc_root: Path, 
        gold_root: Path, 
        disc_format: str,
        gold_format: str,
    ):
        self.disc_root = disc_root
        self.gold_root = gold_root
        self.disc_format = disc_format
        self.gold_format = gold_format


    def onset_offset_cluster(
        self, 
        values: Sequence, 
        prev_offset: float
    ) -> Tuple[float, float, int]:
        """Resolve a 2- or 3-element discovered boundary entry into its
        onset, offset, and cluster ID.

        Parameters
        ----------
        values : Sequence
            A single discovered boundary entry, given as either
            ``(onset, offset, cluster)`` or ``(offset, cluster)``.
        prev_offset : float
            The offset of the previous discovered boundary, used as the
            onset when ``values`` has only 2 elements.
    
        Returns
        -------
        onset : float
            The onset time of the current boundary.
        offset : float
            The offset time of the current boundary.
        cluster : int
            The cluster ID for the current boundary.
        """

        if len(values) == 3:
            onset, offset, cluster = values
        elif len(values) == 2:
            offset, cluster = values
            onset = prev_offset
        else:
            raise ValueError(
                f"Expected 2 or 3 values per entry, got {len(values)}: {values}"
            )
        return onset, offset, cluster


    def frames_to_seconds(
        self, fragments: List[Tuple[str, Interval, int]]
    ) -> List[Tuple[str, Interval, int]]:
        """Convert fragment onset/offset values from frame indices to seconds,
        if all values in ``fragments`` are whole numbers.

        Assumes a frame rate of 50 frames per second (20ms per frame). If any
        onset/offset is non-integer, ``fragments`` is assumed to already be in
        seconds and is returned unchanged.

        Parameters
        ----------
        fragments : list of tuple
            Containing the ``speaker``, ``Interval[onset, offset]``, ``cluster``
            for each discovered fragment.

        Returns
        -------
        fragments : list of tuple
            The input fragments, with onset/offset converted to seconds if they
            were originally frame indices.
        """

        if not fragments:
            return fragments

        all_integer = all(
            float(interval.begin).is_integer() and float(interval.end).is_integer()
            for _, interval, _ in fragments
        )
        if not all_integer:
            return fragments

        return [
            (speaker, Interval(interval.begin / 50, interval.end / 50), cluster)
            for speaker, interval, cluster in fragments
        ]


    def parse_txt_discovery(self) -> Dict[str, List[Tuple[float, float, int]]]:
        """Parse a single ``self.disc_root`` .txt discovered file into a
        speaker-fragments mapping, in file order.

        Returns
        -------
        speaker_fragments : dict of {str: list of tuple}
            Keyed by speaker. Each value is a list of
            ``(start_time, end_time, cluster)`` tuples, one per discovered
            fragment for that speaker.

        Notes
        -----
        Expected line formats in the discovery file:
            - 2 tokens (``label cluster``): sets the active cluster
            for subsequent fragments.
            - 3 tokens (``speaker start_time end_time``): emits a fragment
            using the currently active cluster.
        """

        speaker_fragments = defaultdict(list)
        cluster = None

        with open(self.disc_root, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 3:
                    speaker, start_time, end_time = parts
                    speaker_parts = speaker.split("_")
                    if len(speaker_parts) > 2:
                        speaker = "_".join(speaker_parts[:-2])
                    if cluster is not None:
                        speaker_fragments[speaker].append(
                            (float(start_time), float(end_time), int(cluster))
                        )
                elif len(parts) == 2:
                    cluster = parts[1].split(":")[0] if ":" in parts[1] else parts[1]

        return speaker_fragments


    def disc_to_intervals(self) -> List[Tuple[str, Interval, int]]:
        """Reads the discovered lexicon units.

        Returns
        -------
        fragments : list of tuple
            Containing the ``speaker``, ``Interval[onset, offset]``, ``cluster`` 
            information for each discovered unit (segment and cluster assignment).
        """

        fragments = []

        if self.disc_format == ".list":
            files = self.disc_root.rglob("**/*.list")
            list_fragments = []
            for file in files:
                speaker = file.stem
                start_time = 0.0
                with open(file, "r") as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) not in (2, 3):
                            continue  # skip malformed/empty lines

                        onset, offset, cluster = self.onset_offset_cluster(
                            parts, start_time
                        )

                        if cluster is None or "None" in str(cluster):
                            # skip silence-type entries, but keep offset chain
                            start_time = float(offset)
                            continue

                        list_fragments.append(
                            (
                                speaker, 
                                Interval(float(onset), float(offset)), 
                                int(cluster)
                            )
                        )
                        start_time = float(offset)

            fragments.extend(self.frames_to_seconds(list_fragments))

        elif self.disc_format == ".npy":
            files = self.disc_root.rglob("**/*.npy")
            npy_fragments = []
            for file in files:
                segments = np.load(file, allow_pickle=True)
                speaker = file.stem
                start_time = 0.0
                for row in segments:
                    onset, offset, cluster = self.onset_offset_cluster(
                        tuple(row), start_time
                    )
                    npy_fragments.append(
                        (
                            speaker, 
                            Interval(float(onset), float(offset)), 
                            int(cluster)
                        )
                    )
                    start_time = float(offset)

            fragments.extend(self.frames_to_seconds(npy_fragments))

        elif self.disc_format == ".txt":
            for speaker, frags in self.parse_txt_discovery().items():
                for start_time, end_time, cluster in frags:
                    fragments.append(
                        (speaker, Interval(start_time, end_time), cluster)
                    )
                        
        else:
            raise ValueError("Discovered directory format unsupported.")

        if len(fragments) == 0:
            raise RuntimeError("No discovered fragments found. Please check input parameters.")

        return fragments

    
    def gold_to_grids(self) -> Tuple[Dict[str, textgrids.TextGrid], float, int]:
        """Reads the gold alignments into a ``textgrid.TextGrid`` format.

        Returns
        -------
        grids : dict of {str: textgrids.TextGrid}
            Containing a ``textgrid.TextGrid`` for each utterance, keyed by 
            file stem.
        duration : float
            The duration of the speech data in seconds.
        num_gold_phones : int
            The number of phone tokens in the speech data.
        """

        grids = {}
        duration = 0.0
        num_gold_phones = 0
        files = self.gold_root.rglob("**/*" + self.gold_format)
        for file in files:
            if self.gold_format == ".TextGrid":
                grids[file.stem] = textgrids.TextGrid(file)
                duration += grids[file.stem].xmax - grids[file.stem].xmin
                num_gold_phones += sum(
                    [
                        1 for interval in grids[file.stem]["phones"]
                        if interval.text.lower() not in ["sil", "spn", "sp", ""]
                    ]
                )
                
            elif self.gold_format == ".txt": # ZRC format
                with open(file, "r") as f:
                    if file.stem not in grids: # Create a new TextGrid for each file
                        grids[file.stem] = textgrids.TextGrid()
                    if "phone" in str(file):
                        for line in f:
                            line = line.split()
                            onset = float(line[0])
                            offset = float(line[1])
                            text = str(line[2])
                            grids[file.stem]["phones"].append(
                                textgrids.Interval(text, onset, offset)
                            )
                            if text.lower() not in ["sil", "spn", "sp", ""]:
                                num_gold_phones += 1

                    elif "word" in str(file):
                        for line in f:
                            line = line.split()
                            onset = float(line[0])
                            offset = float(line[1])
                            text = str(line[2])
                            grids[file.stem]["words"].append(
                                textgrids.Interval(text, onset, offset)
                            )
                            duration += offset - onset
                    else:
                        raise ValueError("Text files must be saved in 'phone' and 'word directories.")
                    
            else:
                raise ValueError("Gold directory format unsupported.")

        return grids, duration, num_gold_phones


    def extract_boundaries(self, file_seg: Path) -> List[float]:
        """Extract discovered offset boundaries for an utterance saved in
        ``file_seg``. The file format can be .list or .npy and each file may
        contain either (onset, offset, cluster), (offset, cluster), or (offset).

        Parameters
        ----------
        file_seg : Path
            Path to the discovered boundary file for the current utterance.
    
        Returns
        -------
        boundaries : list of float
            Discovered boundary offsets in seconds.
        """

        boundaries = []
        prev_offset = 0.0

        if self.disc_format == ".npy":
            rows = np.load(file_seg, allow_pickle=True)
        else:
            with open(file_seg, "r") as f:
                rows = [line.strip().split() for line in f if line.strip()]

        for row in rows:
            if len(row) == 1: # no cluster
                offset = float(row[0])
            else:
                _, offset, _ = self.onset_offset_cluster(row, prev_offset)
                offset = float(offset)

            boundaries.append(offset)
            prev_offset = offset

        if boundaries and boundaries[0] == 0.0:
            boundaries = boundaries[1:]

        return boundaries


    def disc_gold_to_pairs(
        self,
        gold_type: str,
        ms_per_frame: int,
        tolerance: Union[int, float],
        frames: bool,
        split_utt: bool
    ) -> Tuple[List[List[Union[int, float]]], List[List[Union[int, float]]]]:
        """Pair discovered segmentation boundaries with gold reference boundaries,
        per utterance, across all files in ``self.disc_root`` and ``self.gold_root``.

        Parameters
        ----------
        gold_type : str
            The ``textgrids.TextGrid`` tier to use for the gold units (e.g.
            ``"words"`` or ``"phones"``). Must be ``"words"`` if
            ``self.gold_format == '.txt'``.
        ms_per_frame : int or float
            Milliseconds per frame, used to convert boundaries to frame indices
            when ``frames`` is ``True``.
        tolerance : int or float
            The number of frames or seconds within which a discovered boundary
            can hit a gold boundary. If ``int``, interpreted as number of
            frames; if ``float``, interpreted as number of seconds.
        frames : bool
            If ``True``, convert both discovered and gold boundaries to frame
            indices (using ``ms_per_frame``) before pairing.
        split_utt : bool
            If ``True``, split each utterance's boundaries around silences. 
            If ``False``, use the entire utterance in the evaluation.

        Returns
        -------
        seg_list : list of list
            Discovered boundaries, one sub-list per utterance (or per utterance
            if ``split_utt`` is ``False``).
        ref_list : list of list
            Gold boundaries, one sub-list per utterance (or per utterance if
            ``split_utt`` is ``False``), aligned with ``seg_list``.
        """
        
        files_ref = sorted(self.gold_root.rglob(f'**/*{self.gold_format}'))

        if self.disc_format == ".txt":
            speaker_fragments = self.parse_txt_discovery()
            files_seg = []
            for ref_file in files_ref:
                if ref_file.stem in speaker_fragments:
                    bounds = [b for _, b, _ in speaker_fragments[ref_file.stem]]
                    files_seg.append(bounds)
                else:
                    raise ValueError("Discovered and gold files not matching.")
        else:
            files_seg = sorted(self.disc_root.rglob(f"**/*{self.disc_format}"))
    
            assert len(files_seg) == len(files_ref)
            for sp, rp in zip(files_seg, files_ref):
                assert sp.stem == rp.stem
    
        seg_list, ref_list = [], []
        for file_seg, file_ref in zip(tqdm(files_seg), files_ref):
            if self.disc_format == ".txt":
                seg_utt = file_seg
            else:
                seg_utt = self.extract_boundaries(file_seg)
    
            if self.gold_format == '.TextGrid':
                tg = textgrids.TextGrid(file_ref)[gold_type]
            elif self.gold_format == '.txt':
                assert gold_type == "words"
                tg = textgrids.TextGrid()
                with open(file_ref, 'r') as f:
                    for line in f:
                        line = line.split()
                        tg.append(
                            textgrids.Interval(
                                line[2], 
                                float(line[0]), 
                                float(line[1])
                            )
                        )
    
            if frames:
                seg_utt = get_frame_num(
                    np.array(seg_utt), ms_per_frame=ms_per_frame
                ).tolist()
                tg = [textgrids.Interval(
                    text=interval.text,
                    xmin=get_frame_num(
                        interval.xmin, 
                        ms_per_frame=ms_per_frame
                    ).item(),
                    xmax=get_frame_num(
                        interval.xmax, 
                        ms_per_frame=ms_per_frame
                    ).item()
                ) for interval in tg]
            
            if split_utt:
                seg_utt, ref_utt = split_utterance(
                    seg_utt, tg, tolerance=tolerance
                )
                seg_list.extend(seg_utt)
                ref_list.extend(ref_utt)
            else:
                seg_list.append(seg_utt)
                ref_list.append([interval.xmax for interval in tg])

        return seg_list, ref_list