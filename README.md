# Unit Discovery Evaluation

This repository evaluates unit discovery metrics for the segmentation and clustering of speech. The metrics covered are: Boundary Precision, Boundary Recall, Boundary F1-Score, Token Precision, Token Recall, Token F1-Score, Over-Segmentation, R-Value, normalized edit distance (NED), coverage, and type scores. Both scripts can be called as main methods, where segmentation and alignment files must be included, alternatively the script's functions can be called from within another scripts by supplying the functions with the correct variables (as noted in the function headers).

## Scripts

Note: all segmentation, alignment, and VAD files are saved in seconds and all file names containing time stamps are denoted in milliseconds.

### Boundary Evaluation

Script name: `boundary_eval.py`

This script evaluates boundary metrics namely: Boundary Precision, Boundary Recall, Boundary F1-Score, Token Precision, Token Recall, Token F1-Score, Over-Segmentation, and R-Value.

**Example Usage**

    python3 boundary_eval.py path/to/segment/files path/to/alignment/files --alignment_format=.TextGrid --alignment_type=words --ms_per_frame=20 --tolerance=1 --split_utterances=False --strict=True

The `alignment_format` argument specifies the extension of the alignment files (options: `.TextGrid`, or `.txt`).
The `alignment_type` argument specifies the level of transcriptions to use (options: `words`, `syllables`, or `phones`).
The `frames_per_ms` argument specifies the number of milliseconds contained in one frame of encoded audio.
The `tolerance` argument specifies the number of frames (to both sides) that the hypothesized boundary can be from the ground truth boundary to still count. If `int`, interpreted as number of frames; if `float`, interpreted as number of seconds.
The `split_utterances` argument determines if there is evaluated only between silences in the reference or not.
The `strict` argument determines if the boundary hit count is strict or lenient as described by D. Harwath in the following paper [https://ieeexplore.ieee.org/abstract/document/10022827](https://ieeexplore.ieee.org/abstract/document/10022827), default `True`.

The input file format is a `.list` file containing offset boundaries (and optional space-separated class assignment values) with each new value (or pair of values) on a new line.

### Cluster Evaluation

Python script name: clust_eval.py

This script evaluates clustering metrics namely: NED (normalized, reverse and accuracy versions), PER (normalized, reverse and accuracy versions), bitrate, coverage, and type scores.

**Example Usage**

    python3 clust_eval.py path/to/alignment/files path/to/output --group_by=words --alignment_format=.TextGrid --disc_format=.list --print_clean

with the same argument definitions as above.
The `group_by` argument specifies the level of transcriptions used for reverse metrics (options: `words` or `syllables`).
The `print_clean` argument controls the format in which the resulting evaluation is printed to the console.

The input file format can either be a `.list` file containing boundaries and a space-separated class assignment value with each new pair of values on a new line similar to above or a single `.txt` file as used in ZeroSpeech evaluation [https://zerospeech.com/challenge_archive/2017/tasks/], where `--disc_format` specifies this.

## Contributors

- [Simon Malan](https://scholar.google.com/citations?user=rxKKwFAAAAAJ&hl=en)
- [Benjamin van Niekerk](https://scholar.google.com/citations?user=zCokvy8AAAAJ&hl=en&oi=ao)
- [Herman Kamper](https://www.kamperh.com/)
- [Danel Slabbert](https://scholar.google.com/citations?user=himAm2QAAAAJ&hl=en&oi=sra)
