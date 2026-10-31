from functools import lru_cache
from langchain.chat_models import init_chat_model
from research_agent.config import settings

_SPEC_BY_ROLE = {
    "planner": settings.model_planner,
    "researcher": settings.model_researcher,
    "synthesizer": settings.model_synthesizer,
    "critic": settings.model_critic,
    "judge": settings.model_judge,
}


@lru_cache(maxsize=None)
def get_model(role: str):
    spec = _SPEC_BY_ROLE[role]
    options = {
        "temperature": settings.temperature,
        "timeout": settings.model_timeout_seconds,
        "max_retries": 0,  # Graph policy owns transport retries; avoid nested retry storms.
    }
    if "deepseek" in spec.lower():
        options["extra_body"] = {"thinking": {"type": "disabled"}}
    return init_chat_model(spec, **options)
