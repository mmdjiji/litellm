"""
Sk Vertex AI provider — wraps the custom Sk endpoint that uses the same
request/response body format as Google Vertex AI generateContent, but with:
  - Custom base URL  (api_base / SK_VERTEX_AI_API_BASE)
  - API-key query-param auth  (?key=…)  instead of Google OAuth Bearer token
  - Extra header: X-Vertex-AI-LLM-Shared-Request-Type: spillover

Usage (via LiteLLM):
    litellm.completion(
        model="sk_vertex_ai/gemini-3-flash-preview",
        messages=[{"role": "user", "content": "hello"}],
        api_key="sk-testkey123",
        api_base="https://sk.example.com/v1",
        vertex_project="project-name",
        vertex_location="us-central1",
    )
"""

from typing import Callable, Literal, Optional, Tuple, Union

import httpx

from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES
from litellm.types.utils import ModelResponse

from ..gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig, VertexLLM

SK_SPILLOVER_HEADER = "X-Vertex-AI-LLM-Shared-Request-Type"
SK_SPILLOVER_VALUE = "spillover"


class SkVertexAIConfig(VertexLLM):
    """
    LiteLLM provider for the Sk custom Vertex-AI-compatible endpoint.

    Inherits all HTTP dispatch logic from VertexLLM, but overrides:
      • _ensure_access_token  — no Google OAuth needed
      • _get_token_and_url    — custom base URL + ?key= query-param auth
      • completion / async_completion — inject spillover header via extra_headers
    """

    # ------------------------------------------------------------------ #
    #  Auth helpers                                                        #
    # ------------------------------------------------------------------ #

    def _ensure_access_token(
        self,
        credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        project_id: Optional[str],
        custom_llm_provider: Literal[  # type: ignore[override]
            "vertex_ai", "vertex_ai_beta", "gemini"
        ],
    ) -> Tuple[str, str]:
        """Sk uses API-key auth in the URL — no Google OAuth token needed."""
        return "", project_id or ""

    async def _ensure_access_token_async(
        self,
        credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        project_id: Optional[str],
        custom_llm_provider: Literal[  # type: ignore[override]
            "vertex_ai", "vertex_ai_beta", "gemini"
        ],
    ) -> Tuple[str, str]:
        """Async version — no OAuth token needed."""
        return "", project_id or ""

    # ------------------------------------------------------------------ #
    #  URL building                                                        #
    # ------------------------------------------------------------------ #

    def _get_token_and_url(  # type: ignore[override]
        self,
        model: str,
        auth_header: Optional[str],
        gemini_api_key: Optional[str],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        stream: Optional[bool],
        custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
        api_base: Optional[str],
        should_use_v1beta1_features: Optional[bool] = False,
        mode: str = "chat",
        use_psc_endpoint_format: bool = False,
    ) -> Tuple[Optional[str], str]:
        """
        Build the Sk endpoint URL:
          {api_base}/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:{op}?key={api_key}

        Auth is embedded as a query param — no Bearer token header returned.
        """
        if not api_base:
            raise ValueError(
                "api_base is required for sk_vertex_ai. "
                "Pass api_base= or set SK_VERTEX_AI_API_BASE."
            )
        if not gemini_api_key:
            raise ValueError(
                "api_key is required for sk_vertex_ai. "
                "Pass api_key= or set SK_VERTEX_AI_API_KEY."
            )
        if not vertex_project:
            raise ValueError(
                "vertex_project is required for sk_vertex_ai. "
                "Pass vertex_project= or set VERTEXAI_PROJECT."
            )
        if not vertex_location:
            raise ValueError(
                "vertex_location is required for sk_vertex_ai. "
                "Pass vertex_location= or set VERTEXAI_LOCATION."
            )

        # Remove routing prefixes (e.g. "bge/", "gemma/") from the model name
        model_name = VertexGeminiConfig.get_model_for_vertex_ai_url(model=model)

        if mode == "chat":
            op = "streamGenerateContent" if stream else "generateContent"
        elif mode == "count_tokens":
            op = "countTokens"
        else:
            op = "generateContent"

        url = (
            f"{api_base.rstrip('/')}/v1/projects/{vertex_project}"
            f"/locations/{vertex_location}/publishers/google/models"
            f"/{model_name}:{op}?key={gemini_api_key}"
        )

        if stream and mode == "chat":
            url += "&alt=sse"

        # auth is in the URL query param — no Authorization header
        return None, url

    # ------------------------------------------------------------------ #
    #  Completion entry-points                                             #
    # ------------------------------------------------------------------ #

    def completion(  # type: ignore[override]
        self,
        model: str,
        messages: list,
        model_response: ModelResponse,
        print_verbose: Callable,
        custom_llm_provider: str,
        encoding,
        logging_obj,
        optional_params: dict,
        acompletion: bool,
        timeout: Optional[Union[float, httpx.Timeout]],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        gemini_api_key: Optional[str],
        litellm_params: dict,
        logger_fn=None,
        extra_headers: Optional[dict] = None,
        client=None,
        api_base: Optional[str] = None,
    ):
        """
        Dispatch to the parent VertexLLM.completion() using "vertex_ai_beta" as
        the internal provider so body transformation uses Vertex AI generateContent
        format.  The spillover header is injected via extra_headers.

        Our overrides of _ensure_access_token and _get_token_and_url transparently
        supply the Sk-specific URL and key-based auth.
        """
        sk_headers = {
            **(extra_headers or {}),
            SK_SPILLOVER_HEADER: SK_SPILLOVER_VALUE,
        }
        return super().completion(
            model=model,
            messages=messages,
            model_response=model_response,
            print_verbose=print_verbose,
            custom_llm_provider="vertex_ai_beta",  # type: ignore[arg-type]
            encoding=encoding,
            logging_obj=logging_obj,
            optional_params=optional_params,
            acompletion=acompletion,
            timeout=timeout,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_credentials=vertex_credentials,
            gemini_api_key=gemini_api_key,
            litellm_params=litellm_params,
            logger_fn=logger_fn,
            extra_headers=sk_headers,
            client=client,
            api_base=api_base,
        )


sk_vertex_ai_chat_completion = SkVertexAIConfig()
