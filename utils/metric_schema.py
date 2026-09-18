import json
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np

@dataclass
class p_r_f1:
    precision: float
    recall: float
    f1: float


@dataclass
class bounds:
    precision: float
    recall: float
    f1: float
    os: float
    r_value: float


@dataclass
class fwd_inv_f1:
    forward: float
    inverse: float
    f1: float


@dataclass
class fwd_inv_d:
    forward: float
    inverse: float
    eucl_dist: float


@dataclass
class v_m:
    homogeneity: float
    completeness: float
    v_measure: float


@dataclass
class mut_inf:
    mi: float
    nmi: float
    class_nmi: float
    cluster_nmi: float


@dataclass
class clss:
    class_type: str
    num_classes: int
    num_singleton_classes: int


@dataclass
class clust:
    num_clusters: int
    num_singleton_clusters: int
    clust_mean_size: float
    clust_std_size: float
    clust_median_size: int
    clust_max_size: int
    clust_min_size: int


@dataclass
class EvaluationResults:
    boundaries: bounds | None = None
    token_boundaries: p_r_f1 | None = None
    num_disc_units: int | None = None
    classes: clss | None = None
    clusters: clust | None = None
    coverage: float | None = None
    bitrate: float | None = None
    original_nes: fwd_inv_f1 | None = None
    weighted_nes: fwd_inv_f1 | None = None
    pacc: fwd_inv_d | None = None
    purity: fwd_inv_f1 | None = None
    v_measure: v_m | None = None
    mutual_info: mut_inf | None = None
    types: p_r_f1 | None = None


class MetricEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


def merge_results(res, prev_res):
    if prev_res is None:
        return res

    merged = dict(res)
    for k, v in merged.items():
        if v is None and prev_res.get(k) is not None:
            merged[k] = prev_res[k]
    return merged


def save_json(name, out_name, results, prev_results):
    if prev_results is None:
        output_dict = {}
    else:
        with open(prev_results, 'r') as file:
            output_dict = json.load(file)
        Path(prev_results).unlink()

    res = asdict(results)
    prev_res = output_dict.get(name)
    output_dict[name] = merge_results(res, prev_res)

    out_path = Path(out_name).with_suffix(".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2, sort_keys=False, cls=MetricEncoder)


def print_result(label, value, percentage=True):
    if len(label) > 25:
        label_len = 45
    else:
        label_len = 25

    if isinstance(value, float):
        if percentage:
            print(f"{label:<{label_len}}:\t{value*100:.2f}")
        else:
            print(f"{label:<{label_len}}:\t{value:.2f}")
    elif isinstance(value, str):
        print(f"{label:<{label_len}}:\t{value}")
    else:
        print(f"{label:<{label_len}}:\t{value:,}")


def print_lex_results(name, results):
    print(f"\n~~~ Summarized evaluation results for {name} ~~~")
    print_result("Bitrate", results.bitrate, percentage=False)
    print_result("NES", results.original_nes.forward)
    print_result("Inverse NES", results.original_nes.inverse)
    print_result("F1 NES", results.original_nes.f1)
    print_result("Weighted NES", results.weighted_nes.forward)
    print_result("Inverse Weighted NES", results.weighted_nes.inverse)
    print_result("F1 Weighted NES", results.weighted_nes.f1)
    print_result("PAcc", results.pacc.forward)
    print_result("Inverse PAcc", results.pacc.inverse)
    print_result("Eucl. Dist. PAcc", results.pacc.eucl_dist)

    print("\n~~~ Lexicon statistics ~~~")
    print_result("# discovered units", results.num_disc_units)
    print_result(f"# {results.classes.class_type} classes", results.classes.num_classes)
    print_result(f"# singleton classes", results.classes.num_singleton_classes)
    print_result("# clusters", results.clusters.num_clusters)
    print_result("# singleton clusters", results.clusters.num_singleton_clusters)
    print_result("Mean (+-std) cluster size", f"{round(results.clusters.clust_mean_size, 2)} (+-{round(results.clusters.clust_std_size, 2)})")
    print_result("Median cluster size", results.clusters.clust_median_size)
    print_result("Max cluster size", results.clusters.clust_max_size)
    print_result("Min cluster size", results.clusters.clust_min_size)