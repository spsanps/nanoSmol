"""Prompt templates shared by the NanoEval runners."""

MMLU_ZERO_SHOT = """\
You are given a multiple-choice question. Choose the correct answer.

Question:
{question}

Options:
A. {A}
B. {B}
C. {C}
D. {D}

Answer:"""

HELLASWAG_ZERO_SHOT = """\
Choose the ending that best completes the context.

Context:
{context}

Endings:
A. {A}
B. {B}
C. {C}
D. {D}

Answer:"""

MMMU_VQA_ZERO_SHOT = """\
You will see one or more images and a question with answer options. Select the option that best answers the question.

Question:
{question}

Options:
{options_block}

Answer:"""

__all__ = [
    "HELLASWAG_ZERO_SHOT",
    "MMLU_ZERO_SHOT",
    "MMMU_VQA_ZERO_SHOT",
]
