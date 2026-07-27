"""Compatibility helpers for model-provider-specific API behavior."""


def get_model_compatibility_config(model_name: str) -> dict:
    """Return provider-specific settings needed by OpenAI-compatible models."""
    if "deepseek-v4" in model_name.lower():
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    return {}


def get_structured_output_config(model_name: str) -> dict:
    """Override structured output only when the provider requires it."""
    if "deepseek-v4" in model_name.lower():
        return {"method": "function_calling"}
    return {}
