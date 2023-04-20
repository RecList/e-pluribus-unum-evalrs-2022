# e-pluribus-unum-evalrs-2022
Repo containing code and data for the analysis presented in: "E Pluribus Unum: Guidelines on Multi-Objective Evaluation of Recommender Systems"

## The Past: EvalRS 2022, Multi-Objective Evaluation in Practice

Clone this repository and execute the following code from the root of the project to replicate the original evaluation protocol in EvalRS 2022.

## Parsing the submissions

```python
import pandas as pd
data = pd.read_parquet("./data/evalrs_2022_submissions.parquet.snappy")
```

MRED is an equality difference measure and hence has to be minimized.
Since the remaining evaluation metrics need to be maximized, following Eq. 1 in the paper, **we report here the negated MRED scores**.
As a result, the aggregated score has to be maximized.

## Computing Stage 1 Scores

The score assigned to the first stage submission is a simple average across all metrics:

```python

METRICS = [
    "hit_rate",
    "mrr",
    "mred_country",
    "mred_track_popularity",
    "mred_user_activity",
    "mred_artist_popularity",
    "mred_gender",
    "being_less_wrong",
    "latent_diversity",
]

def get_score_stage_one(data: pd.DataFrame):
    return data[METRICS].mean(axis=1)

data["score_1"] = get_score_stage_one(data)
```

## Computing Stage 2 Scores

To compute the final score we 

1. picked the best submission for each team in phase 1;
2. filtered our submissions with a hit-rate threshold;
3. computed min-max normalization using a CBOW baseline (min) and the per-metric best (max). For your convenience, we added hardcoded metric values in the code below for both the baseline and the best results;
4. aggregated the metrics with a weighted average. 

Use the following code to replicate our process.

```python
from dataclasses import dataclass

@dataclass
class PhaseOne:

    @property
    def baseline(self):
        return self._CBOW_SCORES

    @property
    def best(self):
        return self._BEST_SCORE_P1

    HR_THRESHOLD = 0.015  # ~ 20% below CBOW HIT RATE

    # scores of our CBOW baseline 
    _CBOW_SCORES = {
        "hit_rate": 0.018763,
        "mrr": 0.001654,
        "mred_country": -0.006944,
        "mred_user_activity": -0.012460,
        "mred_track_popularity": -0.006816,
        "mred_artist_popularity": -0.003915,
        "mred_gender": -0.004354,
        "being_less_wrong": 0.2744871, # Original score (0.322926) decreased by 15%
        "latent_diversity": -0.324706
    }

    # best per-metric score from phase 1, considering only each team best submission
    _BEST_SCORE_P1 = {
        "hit_rate": 0.264642,
        "mrr": 0.067493,
        "mred_country": -0.004490,
        "mred_user_activity": -0.006922,
        "mred_track_popularity": -0.005865,
        "mred_artist_popularity": -0.003623,
        "mred_gender": -0.000032,
        "being_less_wrong": 0.40635,
        "latent_diversity": -0.202812
    }


def get_score_stage_two(row):  
  
    reference = PhaseOne()

    # Check if submission meets minimum reqs
    if row["hit_rate"] < reference.HR_THRESHOLD:
        return -100.0

    normalized_scores = dict()
    for test in METRICS:
        normalized_scores[test] = (
            row[test] - reference.baseline[test]
        ) / (reference.best[test] - reference.baseline[test])

    # Computing meta-scores
    # Performance
    ms_perf = (normalized_scores["hit_rate"] + normalized_scores["mrr"]) / 2
    
    #Fairness / Slicing
    ms_fair = (
        normalized_scores["mred_country"] +
        normalized_scores["mred_user_activity"] +
        normalized_scores["mred_track_popularity"] +
        normalized_scores["mred_artist_popularity"] +
        normalized_scores["mred_gender"]
    ) / 5
    
    # Behavioral
    ms_behav = (
        normalized_scores["being_less_wrong"] + normalized_scores["latent_diversity"]
    ) / 2

    # Meta-scores weights
    w = 1, 1.5, 1.5
    score = (w[0] * ms_perf + w[1] * ms_fair + w[2] * ms_behav) / sum(w)
    
    return score

data["score_2"] = data.apply(get_score_stage_two, axis=1)
```
