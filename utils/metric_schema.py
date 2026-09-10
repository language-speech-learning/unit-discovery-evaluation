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
    gold_unit_nmi: float
    cluster_nmi: float


@dataclass
class mut_inf:
    mi: float
    nmi: float
    gold_unit_nmi: float
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
    num_disc_units: int
    classes: clss
    clusters: clust
    coverage: float
    bitrate: float
    original_nes: fwd_inv_f1
    weighted_nes: fwd_inv_f1
    pacc: fwd_inv_d
    purity: fwd_inv_f1
    v_measure: v_m
    mutual_info: mut_inf
    types: p_r_f1
    boundaries: p_r_f1 | None = None


class MetricEncoder(json.JSONEncoder):
    """Encodes NumPy types, Path objects, and sets into JSON-serializable primitives."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()  # Converts np.int64 -> standard Python int / float
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


def save_json(name, out_name, results):
    output_dict = {}
    output_dict[name] = asdict(results)
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


def print_results(name, results):
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