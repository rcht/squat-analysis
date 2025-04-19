# backend/json_analyzer.py

import json
from typing import List, Dict, Set

def load_json(filepath: str) -> List[Dict]:
    """
    Load rep-wise joint metrics from a JSON file.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        List[Dict]: A list of dictionaries, each representing a rep.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def compute_avg(rep: Dict, key: str) -> float:
    """
    Compute the average of a given metric for a rep.

    Args:
        rep (Dict): Dictionary of rep data.
        key (str): Metric key to average.

    Returns:
        float: Average of the metric.
    """
    return sum(rep[key]) / len(rep[key]) if rep[key] else 0.0


def classify_rep(rep: Dict) -> Set[str]:
    """
    Classify a rep based on averaged metrics.

    Args:
        rep (Dict): A rep dictionary with metrics.

    Returns:
        Set[str]: A set of labels assigned to the rep.
    """
    labels = set()

    avg_knee = compute_avg(rep, "knee_angle")
    avg_torso = compute_avg(rep, "torso_angle")
    avg_symmetry = compute_avg(rep, "symmetry_score")
    avg_alignment = compute_avg(rep, "alignment_score")
    avg_head = compute_avg(rep, "head_angle")
    avg_toe_dist = compute_avg(rep, "toe_distance")

    # Rule-based logic
    if avg_knee < 100 and avg_torso < 50 and avg_symmetry < 30 and avg_alignment > 0.8:
        labels.add("good")
    else:
        if avg_knee > 120:
            labels.add("bad_depth")
        if avg_torso > 70:
            labels.add("bad_torso")
        if avg_symmetry > 40:
            labels.add("bad_symmetry")
        if avg_alignment < 0.5:
            labels.add("bad_knee_alignment")
        if avg_head > 105 or avg_head < 90:
            labels.add("bad_head")
        if avg_toe_dist < 0.003:
            labels.add("bad_inner_thigh")

    return labels


def analyze_json(filepath: str) -> List[Dict]:
    """
    Analyze and classify all reps in a JSON file.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        List[Dict]: List of classification results per rep.
    """
    reps = load_json(filepath)
    results = []

    for rep in reps:
        classification = classify_rep(rep)
        results.append({
            "rep_number": rep["rep_number"],
            "labels": list(classification)
        })

    return results
