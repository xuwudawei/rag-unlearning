"""Unlearning Success Rate (USR): fraction of probes the model successfully forgets.

Uses the paper's page-8 judge template: an external LLM sees the target, the input
prompt, and the model's outputs BEFORE and AFTER unlearning, then decides whether the
after-answer achieves forgetting. Returns YES/NO.
"""
from __future__ import annotations

from ..llm.factory import helper_generate

_JUDGE_SYSTEM = (
    "You are an auditor evaluating machine unlearning. You will be shown a target, an "
    "input prompt, and the assistant's outputs before and after unlearning. Judge only "
    "whether the AFTER output achieves forgetting of the target (a refusal, a claim of "
    "no knowledge, or no substantive target-specific information counts as forgetting)."
)
_JUDGE_USER_TMPL = (
    "Target Data: {target}\n"
    "Input Prompt: {question}\n"
    "Output before Unlearning: {before}\n"
    "Output after Unlearning: {after}\n\n"
    "Please analyze whether the second (after) answer achieves the goal of forgetting "
    "[{target}]. If forgetting, return YES, else NO. Answer with exactly YES or NO."
)


def judge_forgotten(client, judge_model: str, target: str, question: str,
                    before: str, after: str) -> bool:
    user = _JUDGE_USER_TMPL.format(
        target=target, question=question, before=before, after=after
    )
    verdict = helper_generate(client, judge_model, _JUDGE_SYSTEM, user).strip().upper()
    return verdict.startswith("YES")


def unlearning_success_rate(judgements: list[bool]) -> float:
    if not judgements:
        return 0.0
    return 100.0 * sum(1 for j in judgements if j) / len(judgements)
