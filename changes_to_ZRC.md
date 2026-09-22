# Why our evaluation output differs from the ZeroSpeech challenge?

Track 2 of the 2017 ZeroSpeech challenge ([link](https://zerospeech.com/challenge_archive/2017/track2/)) has an evaluation script ([link](https://github.com/zerospeech/zerospeech2020/blob/master/zerospeech2020/evaluation/evaluation_2017_track2.py)) using term-discovery evaluation metrics forked from [bootphon/tdev2](https://github.com/zerospeech/tdev2/tree/master).

These evaluations form the basis of our boundary (`boundary_eval.py`) and lexicon (`clust_eval.py`) evaluation scripts. However, some of our implementation decisions lead to differences compared to the ZeroSpeech implementation, regarding certain metrics and utility functions. These differences are documented below, along with justifications for our updated version.

Note: grouping scores are excluded from our repository based on their limitations regarding computational speed.

## Boundary evaluations

### Gold boundary hit definition

**The ZeroSpeech implementation** maps each predicted boundary to a gold phone boundary and defines a hit (a predicted boundary matching a gold boundary) when this phone boundary is also a word boundary ([details](https://github.com/zerospeech/tdev2/blob/master/tde/measures/boundary.py#L34)).

**Our implementation** (`boundary_eval.py`) defines a boundary hit as an instance when a predicted boundary is within a certain (frame or second) tolerance window of a gold boundary.

We made this decision to (1) enable different boundary tolerances to be used during evaluation (which has become a standard practice in boundary evaluations), and (2) because we feel that mapping a predicted boundary to the a phone boundary can introduce false positives/negatives into the evaluation.

### Token scores

The hit definition above also affects the token scores.

**The ZeroSpeech implementation** defines a token hit when a transcribed predicted segment matches that of the gold token.

**Our implementation** defines a token hit when both the onset and offset boundaries of a predicted segment is within the tolerance of the gold token onset and offset boundaries, respectively.

## Lexicon evaluations

### Normalized edit distance

**The ZeroSpeech implementation** penalizes silence clusters (in [pairwise_ned](https://github.com/zerospeech/tdev2/blob/master/tde/measures/ned.py#L36)). Before pairwise edit distances between phone sequences are calculated, silences in these sequences are removed. When both phone sequences only contain silences (e.g. in a silence cluster), the edit distance between the two, now empty, sequences are set to a maximal distance of 1.0 ([here](https://github.com/zerospeech/tdev2/blob/master/tde/measures/ned.py#L39)).

**Our implementation** disregards any predicted segment that contains only silence. Any silence within a larger phone sequence is also removed from the edit distance calculation.

We made this decision as the ZeroSpeech implementation heavily penalizes silence clusters (all pairwise edit distances will equal 1.0). Furthermore, because NED weighs clusters disproportionally by their size (as shown in our paper), and because silence clusters are usually large, these ''bad'' silence clusters disproportionally worsens the overall NED score. 
We feel that a ''pure'' silence cluster should not be penalized in this way. But, because silence clusters are usually large, we also don't want these clusters to disproportionally improve the NED score. Therefore, we remove all silences when calculating NED (and other clustering metrics), effectively skipping purely silence clusters. 
We argue that what makes these systems useful are their ability to find meaningful speech units. Removing silences ensures that the ability of these systems to segment and cluster speech is the only aspect that is evaluated.

### Type scores

#### Where types can be matched

Token and type scores require the predicted boundaries to find gold types within the speech. Below, we discuss where/how these hits/matches are allowed to happen.

**The ZeroSpeech implementation** defines a type hit as a predicted segment's transcription and location (onset, offset) matching that of the gold type within the same utterance. In the example below, there is one type hit for the gold word ''one''. 

**Our implementation** defines a type hit as a predicted segment's transcription matching that of a gold type anywhere in the speech data. In the example below, there are two hits for the gold words ''one'' and ''per''.

Type score example:
```
Gold:       | One | per | person |
Predicted:  |    |   |  |   |    |  # Transcribed: (one, pe, r, per, son)
```

We do this to relax the strict matching condition to ask whether, in general, the system could find all of the gold types in the speech regardless of where it found it (it could be within a larger gold unit).

#### How type precision is calculated

**The ZeroSpeech implementation** of type precision counts a hit when a predicted segment's phone sequence transcription matches that of the phonetic realization (phone sequence) of the gold type, $type\_hit$. This count is divided by the number of gold types in the data $n\_type$ to get type precision: $\frac{type\_hit}{n\_type}$. This gold type counter $n\_type$, counts the number of unique word-level (not a phoneme sequence) types in the data ([here](https://github.com/zerospeech/tdev2/blob/master/tde/measures/token_type.py#L49)).

**Our implementation** Calculates both a type hit ($type\_hit$) and the number of gold types ($n\_type$) using phonetic realizations.

We believe that $n\_type$ counting the word-level types in the ZeroSpeech implementation is a bug since the precision numerator and denominator are subsequently based on different transcription methods. 
In the example below, both the ZeroSpeech and our implementation would count the number of hits as $type\_hit=3$, but the ZeroSpeech implementation would set $n\_type=2$ (based on the word-level transcriptions), while our implementation would set $n\_type=3$ (based on the phonetic realizations).

```
Gold word unit 1 = (word-level: "the", phonetic-realization: "DH AH")
Gold word unit 2 = (word-level: "the", phonetic-realization: "DH EH")
Gold word unit 3 = (word-level: "cat", phonetic-realization: "K AE T")
Predicted segment transcriptions = ("DH AH", "DH EH", "K AE T")
```

## Utility functions

### Check boundary function

This function is used to decide if a gold phone should be included into the transcription of a discovered unit. This is based on overlap rules, include the gold phone if: (1) it is at least 60 ms long and its overlap with the discovered unit is at least 30 ms long, or (2) it is shorter than 60 ms and its overlap with the discovered unit is at least 50% of the phone duration.

**The ZeroSpeech implementation** rounds each value used in the `check_boundary` condition ([here](https://github.com/zerospeech/tdev2/blob/master/tde/utils.py#L18)) to three decimal points, except for the overlap percentage value ([here](https://github.com/zerospeech/tdev2/blob/master/tde/utils.py#L48)).

**Our implementation** rounds all values to three decimal points.

We do this because floating point errors that are not rounded can cause a phone to be wrongfully included/excluded. Example:
```
Predicted offset boundary = 0.52
Gold phone unit = (0.5, 0.54, "AH") # onset, offset, text

# Without floating-point errors
Gold duration = 0.04 # Therefore check if overlap % is >= 0.5
Predicted unit overlap = 0.02 / 0.04 = 0.5 # No rounding required - phone included

# With floating-point errors
Gold duration = 0.039999
Predicted unit overlap = 0.019999 / 0.039999 = 0.499987 # Rounding required - phone wrongfully excluded
```

These small errors cause our and the ZeroSpeech implementations to provide different transcriptions to some predicted segments. This leads to small differences in the scores of each downstream lexicon-evaluation metric.

## Other useful information

### Bitrate

For non full-coverage systems ($\text{coverage}<99\%$), we divide the entropy by the total duration of the discovered segments to get bitrate in $\text{bits/s}$.

For full-coverage systems ($\text{coverage}\geq99\%$), we divide the entropy by the total duration of the gold alignments to get bitrate in $\text{bits/s}$.

We do this because small differences in full-coverage unsupervised segmentations can lead to slight differences in the total duration of the discovered segments. This duration difference can lead to artifacts in their bitrate scores. Therefore, for full-coverage systems, we rather use the constant total gold duration, which ensures that the bitrate of all full-coverage systems are evaluated equally.