"""Prompt templates shared by the NanoEval runners."""

MMLU_ZERO_SHOT = """\
You are given a multiple-choice question. Choose the correct answer from A, B, C, or D.
Reply using a single capital letter.

Question:
{question}

Options:
A. {A}
B. {B}
C. {C}
D. {D}

Answer:"""

HELLASWAG_ZERO_SHOT = """\
Choose the best ending (A, B, C, or D) that completes the context.
Reply using a single capital letter.

Context:
{context}

Endings:
A. {A}
B. {B}
C. {C}
D. {D}

Answer:"""

MMMU_VQA_ZERO_SHOT = """\
You will see one or more images and a question with answer options.
Reply using a single capital letter from this set: {letters}.

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
