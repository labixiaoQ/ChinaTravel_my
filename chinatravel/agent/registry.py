"""Agent metadata and unified invocation for ChinaTravel runners."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class AgentSpec:
    name: str
    mode: str
    default_llm: str
    max_model_len: int | None = None
    requires_hard_logic: bool = False
    description: str = ""


AGENT_REGISTRY: Dict[str, AgentSpec] = {
    "UrbanTrip": AgentSpec(
        "UrbanTrip", "tpc", "TPCLLM", description="TPC2025 champion symbolic baseline."
    ),
    "TPCAgent": AgentSpec(
        "TPCAgent", "tpc", "TPCLLM", description="Minimal template for a custom TPC agent."
    ),
    "RuleNeSy": AgentSpec(
        "RuleNeSy", "nesy", "deepseek", description="Rule-driven neuro-symbolic agent."
    ),
    "LLMNeSy": AgentSpec(
        "LLMNeSy", "nesy", "deepseek", max_model_len=8192,
        description="LLM-driven neuro-symbolic agent.",
    ),
    "LLM-modulo": AgentSpec(
        "LLM-modulo", "modulo", "deepseek", max_model_len=65536,
        requires_hard_logic=True, description="LLM-Modulo refinement agent.",
    ),
    "Act": AgentSpec("Act", "pure", "deepseek", description="Act prompting baseline."),
    "ReAct": AgentSpec("ReAct", "pure", "deepseek", description="One-shot ReAct baseline."),
    "ReAct0": AgentSpec("ReAct0", "pure", "deepseek", description="Zero-shot ReAct baseline."),
}


def agent_names():
    return tuple(AGENT_REGISTRY)


def get_agent_spec(name: str) -> AgentSpec:
    try:
        return AGENT_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            "Unknown agent {!r}. Available agents: {}".format(
                name, ", ".join(agent_names())
            )
        ) from exc


def run_registered_agent(
    agent: Any,
    spec: AgentSpec,
    query: Dict[str, Any],
    uid: str,
    *,
    include_hard_logic: bool,
    preference_search: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """Normalize the different TPC2025 agent call signatures."""

    if spec.mode == "pure":
        plan_log = agent(query["nature_language"])
        plan = plan_log["ans"]
        if isinstance(plan, str):
            try:
                plan = json.loads(plan)
            except (TypeError, json.JSONDecodeError):
                plan = {"plan": plan}
        if not isinstance(plan, dict):
            plan = {"plan": plan}
        model = getattr(agent, "backbone_llm", None)
        for key in ("input_token_count", "output_token_count", "input_token_maxx"):
            if model is not None and hasattr(model, key):
                plan[key] = getattr(model, key)
        plan["agent_log"] = plan_log["log"]
        return True, plan

    if spec.mode == "modulo":
        return agent.solve(query, prob_idx=uid, oracle_verifier=True)

    if spec.mode == "nesy":
        return agent.run(
            query,
            load_cache=True,
            oralce_translation=include_hard_logic,
            preference_search=preference_search,
        )

    if spec.mode == "tpc":
        return agent.run(
            query,
            prob_idx=uid,
            oralce_translation=include_hard_logic,
        )

    raise ValueError("Unsupported runner mode: {}".format(spec.mode))
