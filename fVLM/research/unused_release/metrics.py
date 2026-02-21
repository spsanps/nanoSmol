"""
NLG metrics for captioning / VQA evaluation.

Wraps pycocoevalcap (CIDEr, BLEU, METEOR) and provides a simple dict-based
interface.  Falls back gracefully if optional dependencies are missing.
"""

from typing import Dict, List


def compute_metrics(
    predictions: List[str],
    references: List[str],
    metrics: List[str] = ("cider", "bleu4", "meteor"),
) -> Dict[str, float]:
    """
    Compute NLG metrics comparing predictions to references.

    Parameters
    ----------
    predictions : list of str
        Generated captions / answers (one per sample).
    references : list of str
        Ground-truth captions / answers (one per sample).
    metrics : tuple of str
        Which metrics to compute.  Options: "cider", "bleu4", "meteor".

    Returns
    -------
    dict
        Metric name -> score.
    """
    assert len(predictions) == len(references), (
        f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
    )

    # Build coco-style format: {image_id: [{"caption": text}]}
    gts = {}
    res = {}
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        gts[i] = [{"caption": ref}]
        res[i] = [{"caption": pred}]

    results = {}

    if "cider" in metrics:
        try:
            from pycocoevalcap.cider.cider import Cider
            scorer = Cider()
            # Cider expects {id: [str]}
            gts_str = {k: [v[0]["caption"]] for k, v in gts.items()}
            res_str = {k: [v[0]["caption"]] for k, v in res.items()}
            score, _ = scorer.compute_score(gts_str, res_str)
            results["cider"] = score
        except ImportError:
            results["cider"] = -1.0

    if "bleu4" in metrics:
        try:
            from pycocoevalcap.bleu.bleu import Bleu
            scorer = Bleu(4)
            gts_str = {k: [v[0]["caption"]] for k, v in gts.items()}
            res_str = {k: [v[0]["caption"]] for k, v in res.items()}
            scores, _ = scorer.compute_score(gts_str, res_str)
            results["bleu4"] = scores[3]  # BLEU-4
        except ImportError:
            results["bleu4"] = -1.0

    if "meteor" in metrics:
        try:
            from pycocoevalcap.meteor.meteor import Meteor
            scorer = Meteor()
            gts_str = {k: [v[0]["caption"]] for k, v in gts.items()}
            res_str = {k: [v[0]["caption"]] for k, v in res.items()}
            score, _ = scorer.compute_score(gts_str, res_str)
            results["meteor"] = score
        except ImportError:
            results["meteor"] = -1.0

    return results


def vqa_accuracy(predictions: List[str], references: List[str]) -> float:
    """
    Simple exact-match accuracy for VQA.

    Normalises both prediction and reference (lowercase, strip) before
    comparing.  Returns a float in [0, 1].
    """
    if not predictions:
        return 0.0
    correct = sum(
        1 for p, r in zip(predictions, references)
        if p.strip().lower() == r.strip().lower()
    )
    return correct / len(predictions)
