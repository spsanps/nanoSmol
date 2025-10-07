"""NanoEval: tiny, config-driven evaluation loops."""

from typing import TYPE_CHECKING

from .config import (
    GenerationConfig,
    HellaSwagRunConfig,
    MMERunConfig,
    MMMUProRunConfig,
    MMLURunConfig,
    ModelConfig,
    ReportConfig,
    ScoringConfig,
    TaskConfig,
    TextVQARunConfig,
    build_model_config,
    build_task_config,
    load_task_config,
)
from .suite import (
    SuiteConfig,
    load_suite_config,
    run_suite,
    run_suite_from_yaml,
)

if TYPE_CHECKING:  # pragma: no cover - imports only needed for type checkers
    from .run_hellaswag import run as _run_hellaswag, run_from_yaml as _run_hellaswag_from_yaml
    from .run_mme import run as _run_mme, run_from_yaml as _run_mme_from_yaml
    from .run_mmlu import run as _run_mmlu, run_from_yaml as _run_mmlu_from_yaml
    from .run_mmmu_pro import run as _run_mmmu_pro, run_from_yaml as _run_mmmu_pro_from_yaml
    from .run_textvqa import run as _run_textvqa, run_from_yaml as _run_textvqa_from_yaml


def run_mmlu(*args, **kwargs):
    from .run_mmlu import run

    return run(*args, **kwargs)


def run_mmlu_from_yaml(*args, **kwargs):
    from .run_mmlu import run_from_yaml

    return run_from_yaml(*args, **kwargs)


def run_hellaswag(*args, **kwargs):
    from .run_hellaswag import run

    return run(*args, **kwargs)


def run_hellaswag_from_yaml(*args, **kwargs):
    from .run_hellaswag import run_from_yaml

    return run_from_yaml(*args, **kwargs)


def run_mmmu_pro(*args, **kwargs):
    from .run_mmmu_pro import run

    return run(*args, **kwargs)


def run_mmmu_pro_from_yaml(*args, **kwargs):
    from .run_mmmu_pro import run_from_yaml

    return run_from_yaml(*args, **kwargs)


def run_textvqa(*args, **kwargs):
    from .run_textvqa import run

    return run(*args, **kwargs)


def run_textvqa_from_yaml(*args, **kwargs):
    from .run_textvqa import run_from_yaml

    return run_from_yaml(*args, **kwargs)


def run_mme(*args, **kwargs):
    from .run_mme import run

    return run(*args, **kwargs)


def run_mme_from_yaml(*args, **kwargs):
    from .run_mme import run_from_yaml

    return run_from_yaml(*args, **kwargs)


__all__ = [
    "ModelConfig",
    "ScoringConfig",
    "ReportConfig",
    "GenerationConfig",
    "MMLURunConfig",
    "HellaSwagRunConfig",
    "MMMUProRunConfig",
    "TextVQARunConfig",
    "MMERunConfig",
    "TaskConfig",
    "build_model_config",
    "build_task_config",
    "load_task_config",
    "run_mmlu",
    "run_mmlu_from_yaml",
    "run_hellaswag",
    "run_hellaswag_from_yaml",
    "run_mmmu_pro",
    "run_mmmu_pro_from_yaml",
    "run_textvqa",
    "run_textvqa_from_yaml",
    "run_mme",
    "run_mme_from_yaml",
    "SuiteConfig",
    "load_suite_config",
    "run_suite",
    "run_suite_from_yaml",
]
