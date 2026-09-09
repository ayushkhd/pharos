"""Constructs a provider by name.

Provider modules are imported lazily so that, for example, a machine
without the OpenAI SDK can still run the CLI-backed providers.
"""

from pharos.providers import base

CODEX_CLI = "codex-cli"
OPENAI = "openai"
CLAUDE_CLI = "claude-cli"
AGY_CLI = "agy-cli"

PROVIDER_NAMES: tuple[str, ...] = (CODEX_CLI, OPENAI, CLAUDE_CLI, AGY_CLI)

# Each provider's sensible default, used when the caller names none.
DEFAULT_MODELS: dict[str, str] = {
    CODEX_CLI: "gpt-5.6-luna",
    OPENAI: "gpt-5.6-luna",
    CLAUDE_CLI: "claude-opus-5",
    AGY_CLI: "gemini-3.1-pro-high",
}


def default_model(name: str) -> str:
  """Returns the default model for a provider.

  Args:
    name: One of ``PROVIDER_NAMES``.

  Raises:
    ValueError: If ``name`` is unknown.
  """
  try:
    return DEFAULT_MODELS[name]
  except KeyError:
    raise ValueError(
        f"unknown provider {name!r}; expected one of {list(PROVIDER_NAMES)}"
    ) from None


def make_provider(
    name: str, model: str | None, effort: str, timeout: float
) -> base.Provider:
  """Builds a provider.

  Args:
    name: One of ``PROVIDER_NAMES``.
    model: Model identifier, or ``None`` for the provider's default.
    effort: Reasoning effort; ignored by providers without the knob.
    timeout: Seconds allowed per call.

  Returns:
    A ready provider.

  Raises:
    ValueError: If ``name`` is unknown.
    base.ProviderUnavailableError: If the provider cannot run here.
  """
  model = model or default_model(name)
  # pylint: disable=import-outside-toplevel
  if name == CODEX_CLI:
    from pharos.providers import codex_cli

    return codex_cli.CodexCliProvider(
        model=model, effort=effort, timeout=timeout
    )
  if name == OPENAI:
    from pharos.providers import openai_api

    return openai_api.OpenAIProvider(
        model=model, effort=effort, timeout=timeout
    )
  if name == CLAUDE_CLI:
    from pharos.providers import claude_cli

    return claude_cli.ClaudeCliProvider(
        model=model, effort=effort, timeout=timeout
    )
  if name == AGY_CLI:
    from pharos.providers import agy_cli

    return agy_cli.AntigravityCliProvider(
        model=model, effort=effort, timeout=timeout
    )
  raise ValueError(
      f"unknown provider {name!r}; expected one of {list(PROVIDER_NAMES)}"
  )
