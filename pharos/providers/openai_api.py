"""OpenAI Responses API provider.

The API route for when no CLI is available or for cross-provider
comparison. The strict ``json_schema`` response format makes the API
enforce the output contract. ``OPENAI_API_KEY`` is read from the process
environment or from a ``.env`` file in the working directory; the key is
never written to any run artifact.
"""

import json
import os
import pathlib
import time
from typing import Any, Protocol

from pharos.providers import base

PROVIDER_NAME = "openai"
DEFAULT_MODEL = "gpt-5.6-luna"
ENV_API_KEY = "OPENAI_API_KEY"
DOTENV_FILE = ".env"
SCHEMA_NAME = "monitor_output"


class ResponsesClient(Protocol):
  """The slice of the OpenAI SDK this provider uses."""

  @property
  def responses(self) -> Any:
    """Returns the Responses API surface."""


def load_dotenv(path: str | pathlib.Path = DOTENV_FILE) -> None:
  """Loads ``KEY=value`` lines from ``path`` into the environment.

  Existing environment variables win. Blank lines and ``#`` comments are
  ignored; surrounding single or double quotes are stripped.

  Args:
    path: The dotenv file; silently skipped if absent.
  """
  path = pathlib.Path(path)
  if not path.is_file():
    return
  for line in path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
      continue
    key, value = line.split("=", 1)
    os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _make_client(timeout: float) -> ResponsesClient:
  """Builds the SDK client.

  Args:
    timeout: Seconds allowed per request.

  Returns:
    A configured client with SDK-level retries disabled; the runner
    owns retry policy.

  Raises:
    base.ProviderUnavailableError: If the SDK is not installed or no API
      key is configured.
  """
  try:
    import openai  # pylint: disable=import-outside-toplevel
  except ImportError as e:
    raise base.ProviderUnavailableError(
        "the openai package is not installed; install pharos[openai]"
    ) from e
  load_dotenv()
  if not os.environ.get(ENV_API_KEY):
    raise base.ProviderUnavailableError(
        f"{ENV_API_KEY} not set; export it or put it in {DOTENV_FILE}"
    )
  client: ResponsesClient = openai.OpenAI(timeout=timeout, max_retries=0)
  return client


class OpenAIProvider:
  """Runs prompts through the Responses API with a strict JSON schema.

  Attributes:
    name: Provider name recorded in manifests.
    model: Model identifier.
    effort: Reasoning effort sent with each request.
    timeout: Seconds allowed per request.
    client: The SDK client (or a test double).
  """

  name = PROVIDER_NAME

  def __init__(
      self,
      model: str = DEFAULT_MODEL,
      effort: str = "low",
      timeout: float = 900,
      client: ResponsesClient | None = None,
  ):
    """Prepares the client.

    Args:
      model: Model identifier.
      effort: Reasoning effort.
      timeout: Seconds allowed per request.
      client: An SDK client to use instead of building one; for tests.

    Raises:
      base.ProviderUnavailableError: If the SDK or the API key is missing.
    """
    self.model = model
    self.effort = effort
    self.timeout = timeout
    self.client = client if client is not None else _make_client(timeout)

  def complete(self, prompt: str, schema: dict[str, Any]) -> base.RawResponse:
    """Runs one prompt.

    Args:
      prompt: The rendered prompt.
      schema: JSON Schema the reply must satisfy.

    Returns:
      The response; failures are recorded on it, never raised.
    """
    started = time.time()
    try:
      reply = self.client.responses.create(
          model=self.model,
          input=[{"role": "user", "content": prompt}],
          reasoning={"effort": self.effort},
          text={
              "format": {
                  "type": "json_schema",
                  "name": SCHEMA_NAME,
                  "strict": True,
                  "schema": schema,
              }
          },
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      # The SDK raises a family of transport, auth and rate-limit errors;
      # a provider must never propagate any of them to the runner.
      kind = (
          base.KIND_TIMEOUT
          if "timeout" in str(e).lower()
          else (base.KIND_PROVIDER)
      )
      return base.RawResponse(duration_s=time.time() - started).fail(
          kind, f"{type(e).__name__}: {str(e)[:800]}"
      )

    usage = getattr(reply, "usage", None)
    response = base.RawResponse(
        duration_s=time.time() - started,
        tokens_used=getattr(usage, "total_tokens", None) if usage else None,
        exit_code=0,
        meta={"response_id": getattr(reply, "id", None)},
    )
    text = getattr(reply, "output_text", "") or ""
    response.text = text
    try:
      decoded = json.loads(text)
    except json.JSONDecodeError as e:
      return response.fail(base.KIND_PARSE, f"json parse failed: {e}")
    if not isinstance(decoded, dict):
      return response.fail(base.KIND_PARSE, "reply is not a JSON object")
    response.parsed = decoded
    return response
