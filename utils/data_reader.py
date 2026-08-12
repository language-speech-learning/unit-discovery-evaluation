from pathlib import Path
from typing import Iterable, Tuple

from textgrid import TextGrid, IntervalTier
from intervaltree import Interval
import numpy as np


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

    # TODO disc_gold_to_pairs

    def disc_to_intervals(self) -> Iterable[Tuple[str, Interval, int]]:
        """Reads the discovered lexicon units.

        Returns
        -------
        fragments : Iterable[Tuple[str, Interval, int]]
            Containing the ``speaker``, ``Interval[onset, offset]``, ``cluster`` 
            information for each discovered unit (segment and cluster assignment).
        """

        fragments = []

        if self.disc_format == ".list":
            files = self.disc_root.rglob("**/*" + ".list")
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
                            fragments.append(
                                (
                                    speaker, 
                                    Interval(float(start_time), float(end_time)), 
                                    int(cluster)
                                )
                            )
                            start_time = float(end_time)

        elif self.disc_format == ".npy":
            files = self.disc_root.rglob("**/*" + ".npy")
            for file in files:
                segments = np.load(file)
                speaker = file.stem
                for start_frame, end_frame, cluster in segments:
                
                    start_time = start_frame / 50.0
                    end_time = end_frame / 50.0
                    fragments.append(
                        (
                            speaker, 
                            Interval(float(start_time), float(end_time)), 
                            int(cluster)
                        )
                    )
        
        elif self.disc_format == ".txt":
            cluster = None
            with open(self.disc_root, "r") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == 3: 
                        speaker, start_time, end_time = parts[0], parts[1], parts[2]
                        speaker_parts = speaker.split("_")
                        if len(speaker_parts) > 2:
                            speaker = "_".join(speaker_parts[:-2]) 
                        if cluster is not None:
                            fragments.append(
                                (speaker, 
                                Interval(float(start_time), float(end_time)), 
                                int(cluster))
                            )
                    elif len(parts) == 2:
                        if ":" in parts[1]: 
                            cluster = parts[1].split(":")[0]
                        else:
                            cluster = parts[1]
        
        else:
            raise ValueError("Discovered directory format unsupported.")
        
        if len(fragments) == 0:
            raise RuntimeError("No discovered fragments found. Please check input parameters.")

        return fragments
    
    def gold_to_grids(self) -> dict:
        """Reads the gold alignments into a TextGrid format.

        Returns
        -------
        grids : dict
            Containing a ``TextGrid`` for each utterance.
        """

        grids = {}
        duration = 0.0
        files = self.gold_root.rglob("**/*" + self.gold_format)
        for file in files:
            if self.gold_format == ".TextGrid":
                grids[file.stem] = TextGrid.fromFile(file)
                duration += grids[file.stem].maxTime - grids[file.stem].minTime
                
            elif self.gold_format == ".txt": # ZRC format
                with open(file, "r") as f:
                    if file.stem not in grids: # Create a new TextGrid for each file
                        grids[file.stem] = TextGrid()
                    if "phone" in str(file):
                        interval_tier = IntervalTier(name="phones")
                    elif "word" in str(file):
                        interval_tier = IntervalTier(name="words")
                    for line in f:
                        line = line.split()
                        onset = float(line[0])
                        offset = float(line[1])
                        interval_tier.add(onset, offset, line[2])
                        duration += offset - onset
                    grids[file.stem].append(interval_tier)
            
            else:
                raise ValueError("Gold directory format unsupported.")
        
        return grids, duration