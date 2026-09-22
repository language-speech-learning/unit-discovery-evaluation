# Unit Discovery Evaluation

[![paper](https://img.shields.io/badge/Paper-Read-%23b31b1b?logo=arXiv&logoColor=%23b31b1b)](https://arxiv.org/abs/2606.06183)

Unsupervised unit discovery systems segment speech into linguistically meaningful units like words or syllables and then cluster these units based on their similarity.

This repository evaluates both the segmentation and lexical quality of such systems. The lexicon evaluation metrics are explained in [our paper](https://arxiv.org/abs/2606.06183) and are shown to provide a fairer evaluation than previous metrics.

## Boundary Evaluation

This script evaluates core boundary and token metrics, including Boundary Precision, Boundary Recall, Boundary F1-Score, Token Precision, Token Recall, Token F1-Score, Over-Segmentation, and R-Value.

Running this script prints the evaluation results to the console and saves a copy of the boundary results in `scores/boundary_scores_<system_name>`. If the current system has already been evaluated by `clust_eval.py`, the boundary and lexicon results are saved in `scores/system_scores_<system_name>`.
### Example Usage

```sh
python3 boundary_eval.py path/to/segment/files path/to/alignment/files \
  --disc_format=.list \
  --gold_format=.TextGrid \
  --gold_type=words \
  --ms_per_frame=20 \
  --tolerance=1 \
  --split_utterances=False \
  --strict=True \
  --system_name=segment_files_stem
```

| Argument | Type / Options | Default | Description |
| :--- | :--- | :--- | :--- |
| `disc_root` | `Path` | Required | Path to the directory containing the discovered segments/boundaries to evaluate. |
| `gold_root` | `Path` | Required | Path to the directory containing the reference alignment files. |
| `--disc_format` | `.list`, `.txt`, `.npy` | `.list` | File extension of the discovered files. |
| `--gold_format` | `.TextGrid`, `.txt` | `.TextGrid` | File extension of the alignment files. |
| `--gold_type` | `words`, `syllables`, `phones` | `words` | The unit level in the transcriptions to use as gold boundaries. |
| `--ms_per_frame` | `int` or `float` | `20` | Number of milliseconds contained within one frame of encoded audio. |
| `--tolerance` | `int` or `float` | `1` | Max allowed distance (frames if `int`, seconds if `float`) for a hypothesized boundary to hit a ground truth boundary. |
| `--split_utterances` | `bool` | `False` | Determines whether evaluation occurs exclusively between silences in the reference. |
| `--strict` | `bool` | `True` | Determines if boundary hit counts are strict or lenient, as defined by [Harwath (2023)](https://ieeexplore.ieee.org/abstract/document/10022827). |
| `--system_name` | `str` | `disc_root.stem` | Name of the results output file. |

>The required file structures are provided at the end of this document.

## Cluster Evaluation

This script evaluates a lexicon using the metrics in [our paper](https://arxiv.org/abs/2606.06183). More details about the metrics and how they differ from the [ZeroSpeech](https://zerospeech.com/challenge_archive/2017/track2/) challenge evaluations can be found in [`changes_to_ZRC.md`](https://github.com/language-speech-learning/unit-discovery-evaluation/blob/main/changes_to_ZRC.md).

Running this script prints a subset of the evaluation results to the console and saves a copy of the lexicon results in `scores/lexicon_scores_<system_name>`. If the current system has already been evaluated by `boundary_eval.py`, the boundary and lexicon results are saved in `scores/system_scores_<system_name>`.

### Example Usage

```sh
python3 clust_eval.py path/to/segment/files path/to/alignment/files \
  --disc_format=.list \
  --gold_format=.TextGrid \
  --gold_type=words \
  --system_name=segment_files_stem
```

| Argument | Type / Options | Default | Description |
| :--- | :--- | :--- | :--- |
| `disc_root` | `Path` | Required | Path to the directory containing the clustered discovered segments to evaluate. |
| `gold_root` | `Path` | Required | Path to the directory containing the reference alignment files. |
| `--disc_format` | `.list`, `.txt`, `.npy` | `.list` | File extension of the discovered files. |
| `--gold_format` | `.TextGrid`, `.txt` | `.TextGrid` | File extension of the alignment files. |
| `--gold_type` | `words`, `syllables`, `disc`| `words` | The unit level in the transcriptions to use as gold types. If `disc`, uses the unique discovered units as types, useful if the discovered segments don't require matching a gold type. |
| `--system_name` | `str` | `disc_root.stem` | Name of the results output file. |

>The required file structures are provided below.

## Discovered Segments File Structure

#### `.list` and `.npy` (matrix) structure (one file per utterance)

- For `boundary_eval.py`, depending on the `tolerance` used, the `<onset>` and `<offset>` values will either be in seconds (`float`) or frames (`int`). `<clusterID>` is optional and is always an integer. 
- For `clust_eval.py`, the `<onset>` and `<offset>` values can be either seconds or frames (since lexicon evaluations are done in seconds, saving your output in seconds will be faster to load). `<clusterID>` is required and is always an integer.

Per file/utterance, any of the following structures are allowed:

```
<onset> <offset> <clusterID>
...
```
```
<offset> <clusterID>
...
```
```
<offset>
...
```

#### `.txt` structure (one file per dataset)

Following the ZeroSpeech challenge, "Class 0" refers to the cluster with ID 0, with all segments contained in this cluster listed below it. All `<start_sec>` and `<end_sec>` values are in seconds.

```
Class 0
<utteranceID> <start_sec> <end_sec>
<utteranceID> <start_sec> <end_sec>

Class 1
<utteranceID> <start_sec> <end_sec>
...
```

## Gold Alignment Files

Alignment files can either be in `.TextGrid` or `.txt` format.

If `.txt` (typically for ZeroSpeech data) is used, ensure that the `path/to/alignment/files` points to the root of the `phone` and `word` directories, each containing the corresponding alignments in seconds. Instructions to preprocess the ZeroSpeech data can be found at [s-malan/data-process](https://github.com/s-malan/data-process).

If `.TextGrid` is used together with LibriSpeech, we include the alignment files in the release of this repository.

## Toy Lexicon Examples

The toy examples used in [*Section VI.B*](https://arxiv.org/abs/2606.06183) of our paper is added in the releases of this repository, in `.npy` format.

## Citation

```
@inproceedings{malan26_eval,
      author={Simon Malan and Danel Slabbert and Herman Kamper},
      title={Revisiting Lexicon Evaluation in Unsupervised Word Discovery}, 
      booktitle={SLT},
      year={2026},
}
```