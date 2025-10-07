"""TextVQA evaluation loop following the NanoEval philosophy."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from datasets import load_dataset
from tqdm import tqdm

from .common import SimpleModel, set_seed
from .config import TextVQARunConfig, load_task_config
from .reporting import ReportWriter

_ARTICLES = {"a", "an", "the"}
_PUNCT_PATTERN = re.compile(r"[^a-z0-9\s]")


def _normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation, and drop articles."""

    lowered = text.lower().strip()
    cleaned = _PUNCT_PATTERN.sub(" ", lowered)
    tokens = [token for token in cleaned.split() if token not in _ARTICLES]
    return " ".join(tokens)


def _coerce_answers(raw: object) -> List[str]:
    """Return the list of ground-truth answers for a TextVQA example."""

    if raw is None:
        return []
    if isinstance(raw, Mapping):
        candidate = raw.get("answer")
        return [str(candidate)] if candidate is not None else []
    answers: List[str] = []
    for item in raw:  # type: ignore[assignment]
        if isinstance(item, Mapping) and "answer" in item:
            answers.append(str(item["answer"]))
        else:
            answers.append(str(item))
    return answers


def _score_prediction(prediction: str, answers: Sequence[str]) -> float:
    """Compute official VQA accuracy using leave-one-out scoring."""

    if not answers:
        return 0.0
    normalized_prediction = _normalize_answer(prediction)
    normalized_answers = [_normalize_answer(answer) for answer in answers]
    per_answer_scores: List[float] = []
    for index in range(len(normalized_answers)):
        others = normalized_answers[:index] + normalized_answers[index + 1 :]
        matches = sum(other == normalized_prediction for other in others)
        per_answer_scores.append(min(matches / 3.0, 1.0))
    return sum(per_answer_scores) / len(per_answer_scores)


def _format_prompt(question: str) -> str:
    return (
        "You are a helpful assistant that answers questions about images.\n"
        f"Question: {question}\n"
        "Answer:"
    )


def run(config: TextVQARunConfig) -> Dict[str, object]:
    """Evaluate TextVQA according to ``config`` and return the summary payload."""

    set_seed(config.scoring.seed)

    model = SimpleModel(config.model)
    dataset = load_dataset("lmms-lab/textvqa", split=config.dataset.split)
    if config.dataset.subset_size is not None:
        dataset = dataset.select(range(min(config.dataset.subset_size, len(dataset))))

    rows: List[Dict[str, object]] = []
    total_score = 0.0

    for example in tqdm(dataset, desc=f"textvqa:{config.dataset.split}"):
        question = str(example.get("question", ""))
        answers = _coerce_answers(example.get("answers"))
        image = example.get("image")
        prompt = _format_prompt(question)
        if model.processor is None:
            prediction = model.generate_text(
                prompt, max_new_tokens=config.generation.max_new_tokens
            )
        else:
            prediction = model.generate_text(
                prompt,
                images=[image] if image is not None else None,
                max_new_tokens=config.generation.max_new_tokens,
            )
        score = _score_prediction(prediction, answers)
        total_score += score
        question_id = (
            example.get("question_id")
            or example.get("questionId")
            or example.get("questionid")
            or example.get("qid")
        )
        image_id = example.get("image_id") or example.get("imageId")
        rows.append(
            {
                "task": "textvqa",
                "question_id": question_id,
                "image_id": image_id,
                "question": question,
                "prediction": prediction,
                "normalized_prediction": _normalize_answer(prediction),
                "answers": answers,
                "score": score,
            }
        )

    total_examples = len(rows)
    accuracy = total_score / max(1, total_examples)

    writer = ReportWriter(config.report, title="TextVQA accuracy")
    writer.write_predictions(rows)
    writer.write_summary(
        {
            "task": "textvqa",
            "accuracy": accuracy,
            "total_examples": total_examples,
            "split": config.dataset.split,
            "subset_size": config.dataset.subset_size,
            "max_new_tokens": config.generation.max_new_tokens,
            "seed": config.scoring.seed,
            "model_id": config.model.model_id,
        }
    )
    writer.write_metrics_table([(config.dataset.split, accuracy)])
    writer.plot_metrics([(config.dataset.split, accuracy)])

    return {"accuracy": accuracy, "total_examples": total_examples}


def run_from_yaml(config_path: Path) -> Dict[str, object]:
    cfg = load_task_config(config_path)
    if not isinstance(cfg, TextVQARunConfig):
        raise TypeError("TextVQA runner received a configuration for a different task")
    return run(cfg)


def _main() -> None:
    config_env = os.environ.get("NANOEVAL_CONFIG")
    if config_env is None:
        raise SystemExit("Set NANOEVAL_CONFIG to the path of a TextVQA config YAML")
    summary = run_from_yaml(Path(config_env))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _main()
