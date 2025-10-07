from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.nanoeval.config import build_task_config  # noqa: E402
from eval.nanoeval.run_mme import _extract_yes_no  # noqa: E402
from eval.nanoeval.run_textvqa import _coerce_answers, _normalize_answer, _score_prediction  # noqa: E402


def test_normalize_answer_strips_articles_and_punctuation():
    assert _normalize_answer("The,  Blue! Car?") == "blue car"


def test_score_prediction_matches_official_rule():
    answers = ["yes"] * 7 + ["no"] * 3
    assert pytest.approx(_score_prediction("Yes", answers), rel=1e-6) == 1.0
    assert pytest.approx(_score_prediction("no", answers), rel=1e-6) == 1.0


def test_score_prediction_returns_thirds_for_partial_agreement():
    answers = ["green"] * 7 + ["blue"] * 2 + ["red"]
    assert pytest.approx(_score_prediction("green", answers), rel=1e-6) == 1.0
    assert pytest.approx(_score_prediction("blue", answers), rel=1e-6) == pytest.approx(2 / 3, rel=1e-6)
    assert pytest.approx(_score_prediction("red", answers), rel=1e-6) == pytest.approx(1 / 3, rel=1e-6)
    assert pytest.approx(_score_prediction("yellow", answers), rel=1e-6) == 0.0


def test_coerce_answers_handles_dict_entries():
    payload = [{"answer": "ten"}, {"answer": "ten"}, "TEN"]
    assert _coerce_answers(payload) == ["ten", "ten", "TEN"]


def test_extract_yes_no_prefers_first_confident_token():
    assert _extract_yes_no("Absolutely yes, no doubt.") == "yes"
    assert _extract_yes_no("Nope, sorry!") == "no"
    assert _extract_yes_no("Maybe") is None


def test_build_task_config_for_textvqa_infers_generation_defaults():
    cfg = build_task_config({
        "task": "textvqa",
        "model": {"model_id": "dummy", "is_vlm": True},
    })
    assert cfg.task == "textvqa"
    assert cfg.generation.max_new_tokens == 32


def test_build_task_config_for_mme_sets_subset_defaults():
    cfg = build_task_config({
        "task": "mme",
        "model": {"model_id": "dummy", "is_vlm": True},
        "dataset": {"split": "test", "subset_size": 5},
        "generation": {"max_new_tokens": 8},
    })
    assert cfg.task == "mme"
    assert cfg.dataset.subset_size == 5
    assert cfg.generation.max_new_tokens == 8
