"""
Unit tests for the sk_vertex_ai custom provider.

Tests cover:
1. URL building (with and without streaming)
2. Header injection (spillover header present, no Bearer token)
3. OpenAI → Vertex AI request body transformation
4. Vertex AI → OpenAI response body transformation
5. Cost calculation via litellm.completion_cost()
6. End-to-end routing through litellm.completion() (mocked HTTP)
"""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm.llms.vertex_ai.sk_endpoint.handler import (
    SK_SPILLOVER_HEADER,
    SK_SPILLOVER_VALUE,
    SkVertexAIConfig,
)
from litellm.types.utils import ModelResponse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

API_BASE = "https://sk.example.com/v1"
API_KEY = "sk-testkey123"
PROJECT = "project-name"
LOCATION = "us-central1"
MODEL = "gemini-3-flash-preview"
PROVIDER_MODEL = f"sk_vertex_ai/{MODEL}"


@pytest.fixture()
def config():
    return SkVertexAIConfig()


# ---------------------------------------------------------------------------
# 1. URL building
# ---------------------------------------------------------------------------


class TestGetTokenAndUrl:
    def test_basic_url(self, config):
        _, url = config._get_token_and_url(
            model=MODEL,
            auth_header=None,
            gemini_api_key=API_KEY,
            vertex_project=PROJECT,
            vertex_location=LOCATION,
            vertex_credentials=None,
            stream=False,
            custom_llm_provider="vertex_ai_beta",  # type: ignore
            api_base=API_BASE,
        )
        assert url.startswith(API_BASE)
        assert f"/v1/projects/{PROJECT}/locations/{LOCATION}" in url
        assert f"/publishers/google/models/{MODEL}:generateContent" in url
        assert f"?key={API_KEY}" in url
        # streaming params should NOT be present
        assert "streamGenerateContent" not in url
        assert "alt=sse" not in url

    def test_streaming_url(self, config):
        _, url = config._get_token_and_url(
            model=MODEL,
            auth_header=None,
            gemini_api_key=API_KEY,
            vertex_project=PROJECT,
            vertex_location=LOCATION,
            vertex_credentials=None,
            stream=True,
            custom_llm_provider="vertex_ai_beta",  # type: ignore
            api_base=API_BASE,
        )
        assert "streamGenerateContent" in url
        assert "alt=sse" in url
        assert f"?key={API_KEY}" in url

    def test_count_tokens_url(self, config):
        _, url = config._get_token_and_url(
            model=MODEL,
            auth_header=None,
            gemini_api_key=API_KEY,
            vertex_project=PROJECT,
            vertex_location=LOCATION,
            vertex_credentials=None,
            stream=False,
            custom_llm_provider="vertex_ai_beta",  # type: ignore
            api_base=API_BASE,
            mode="count_tokens",
        )
        assert "countTokens" in url

    def test_auth_header_is_none(self, config):
        auth_header, _ = config._get_token_and_url(
            model=MODEL,
            auth_header=None,
            gemini_api_key=API_KEY,
            vertex_project=PROJECT,
            vertex_location=LOCATION,
            vertex_credentials=None,
            stream=False,
            custom_llm_provider="vertex_ai_beta",  # type: ignore
            api_base=API_BASE,
        )
        assert auth_header is None, "No Bearer token expected for Sk provider"

    def test_missing_api_base_raises(self, config):
        with pytest.raises(ValueError, match="api_base"):
            config._get_token_and_url(
                model=MODEL,
                auth_header=None,
                gemini_api_key=API_KEY,
                vertex_project=PROJECT,
                vertex_location=LOCATION,
                vertex_credentials=None,
                stream=False,
                custom_llm_provider="vertex_ai_beta",  # type: ignore
                api_base=None,
            )

    def test_missing_api_key_raises(self, config):
        with pytest.raises(ValueError, match="api_key"):
            config._get_token_and_url(
                model=MODEL,
                auth_header=None,
                gemini_api_key=None,
                vertex_project=PROJECT,
                vertex_location=LOCATION,
                vertex_credentials=None,
                stream=False,
                custom_llm_provider="vertex_ai_beta",  # type: ignore
                api_base=API_BASE,
            )

    def test_trailing_slash_stripped_from_api_base(self, config):
        _, url = config._get_token_and_url(
            model=MODEL,
            auth_header=None,
            gemini_api_key=API_KEY,
            vertex_project=PROJECT,
            vertex_location=LOCATION,
            vertex_credentials=None,
            stream=False,
            custom_llm_provider="vertex_ai_beta",  # type: ignore
            api_base=API_BASE + "/",
        )
        # Should not have double slash
        assert "//" not in url.split("://")[1]


# ---------------------------------------------------------------------------
# 2. Headers — spillover header is injected via extra_headers in completion()
# ---------------------------------------------------------------------------


class TestHeaderInjection:
    """
    The spillover header is injected through extra_headers inside completion().
    We verify this by checking that the header appears in the final POST call.
    """

    def _make_mock_httpx_response(self, text: str = "Hi!") -> MagicMock:
        body = {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": text}]},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 2,
                "candidatesTokenCount": 1,
                "totalTokenCount": 3,
            },
            "modelVersion": MODEL,
        }
        mock = MagicMock(spec=httpx.Response)
        mock.status_code = 200
        mock.text = json.dumps(body)
        mock.json.return_value = body
        mock.headers = httpx.Headers({"content-type": "application/json"})
        mock.raise_for_status = MagicMock()
        return mock

    @patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
    def test_spillover_header_in_request(self, mock_post):
        mock_post.return_value = self._make_mock_httpx_response()

        litellm.completion(
            model=PROVIDER_MODEL,
            messages=[{"role": "user", "content": "test"}],
            api_key=API_KEY,
            api_base=API_BASE,
            vertex_project=PROJECT,
            vertex_location=LOCATION,
        )

        called_headers = mock_post.call_args.kwargs.get("headers", {})
        assert called_headers.get(SK_SPILLOVER_HEADER) == SK_SPILLOVER_VALUE

    @patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
    def test_no_bearer_token_in_request(self, mock_post):
        mock_post.return_value = self._make_mock_httpx_response()

        litellm.completion(
            model=PROVIDER_MODEL,
            messages=[{"role": "user", "content": "test"}],
            api_key=API_KEY,
            api_base=API_BASE,
            vertex_project=PROJECT,
            vertex_location=LOCATION,
        )

        called_headers = mock_post.call_args.kwargs.get("headers", {})
        assert "Authorization" not in called_headers

    @patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
    def test_content_type_in_request(self, mock_post):
        mock_post.return_value = self._make_mock_httpx_response()

        litellm.completion(
            model=PROVIDER_MODEL,
            messages=[{"role": "user", "content": "test"}],
            api_key=API_KEY,
            api_base=API_BASE,
            vertex_project=PROJECT,
            vertex_location=LOCATION,
        )

        called_headers = mock_post.call_args.kwargs.get("headers", {})
        assert called_headers.get("Content-Type") == "application/json"

    @patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
    def test_extra_headers_merged(self, mock_post):
        mock_post.return_value = self._make_mock_httpx_response()

        litellm.completion(
            model=PROVIDER_MODEL,
            messages=[{"role": "user", "content": "test"}],
            api_key=API_KEY,
            api_base=API_BASE,
            vertex_project=PROJECT,
            vertex_location=LOCATION,
            extra_headers={"X-Custom-Header": "custom-value"},
        )

        called_headers = mock_post.call_args.kwargs.get("headers", {})
        assert called_headers.get("X-Custom-Header") == "custom-value"
        assert called_headers.get(SK_SPILLOVER_HEADER) == SK_SPILLOVER_VALUE


# ---------------------------------------------------------------------------
# 3. ensure_access_token
# ---------------------------------------------------------------------------


class TestEnsureAccessToken:
    def test_returns_empty_token(self, config):
        token, project = config._ensure_access_token(
            credentials=None,
            project_id=PROJECT,
            custom_llm_provider="vertex_ai_beta",  # type: ignore
        )
        assert token == ""
        assert project == PROJECT

    def test_returns_project_id(self, config):
        _, project = config._ensure_access_token(
            credentials=None,
            project_id="my-project",
            custom_llm_provider="vertex_ai_beta",  # type: ignore
        )
        assert project == "my-project"


# ---------------------------------------------------------------------------
# 4. Response transformation (Vertex AI → OpenAI format)
# ---------------------------------------------------------------------------


class TestResponseTransformation:
    """Test that VertexGeminiConfig.transform_response is reused correctly."""

    def _make_vertex_response(self, text: str = "Hello!") -> dict:
        return {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": text}],
                    },
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 3,
                "totalTokenCount": 8,
            },
            "modelVersion": MODEL,
        }

    def test_transform_response_text(self, config):
        from litellm.types.utils import ModelResponse

        # transform_response lives in VertexGeminiConfig, which is used internally
        from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
            VertexGeminiConfig,
        )

        raw = self._make_vertex_response("Hello, world!")
        raw_json_str = json.dumps(raw)

        # Build a real httpx.Response-like object so json() and text work
        raw_httpx = MagicMock()
        raw_httpx.text = raw_json_str
        raw_httpx.json.return_value = raw
        raw_httpx.headers = httpx.Headers({"content-type": "application/json"})

        logging_mock = MagicMock()
        logging_mock.optional_params = {}
        logging_mock.post_call = MagicMock()

        model_response = ModelResponse(model=MODEL)
        result = VertexGeminiConfig().transform_response(
            model=MODEL,
            raw_response=raw_httpx,
            model_response=model_response,
            logging_obj=logging_mock,
            api_key="",
            request_data={},
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert result.choices[0].message.content == "Hello, world!"
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.total_tokens == 8


# ---------------------------------------------------------------------------
# 5. Cost calculation
# ---------------------------------------------------------------------------


class TestCostCalculation:
    def test_model_in_pricing_db(self):
        """sk_vertex_ai/gemini-3-flash-preview should be in pricing JSON."""
        cost_info = litellm.model_cost.get(PROVIDER_MODEL)
        assert cost_info is not None, f"{PROVIDER_MODEL!r} not in model_prices_and_context_window.json"
        assert "input_cost_per_token" in cost_info
        assert "output_cost_per_token" in cost_info

    def test_completion_cost(self):
        """completion_cost() should return a numeric cost for sk provider."""
        model_response = ModelResponse(
            id="test-id",
            created=0,
            model=PROVIDER_MODEL,
        )
        model_response.usage = MagicMock()
        model_response.usage.prompt_tokens = 100
        model_response.usage.completion_tokens = 50

        cost = litellm.completion_cost(
            completion_response=model_response,
            model=PROVIDER_MODEL,
        )
        assert isinstance(cost, float)
        assert cost > 0


# ---------------------------------------------------------------------------
# 6. End-to-end routing (mocked HTTP call)
# ---------------------------------------------------------------------------


class TestEndToEndRouting:
    """Verify that litellm.completion routes correctly and calls the right URL."""

    def _make_mock_httpx_response(self, text: str = "Hi there!") -> MagicMock:
        vertex_body = {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": text}]},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 4,
                "candidatesTokenCount": 3,
                "totalTokenCount": 7,
            },
            "modelVersion": MODEL,
        }
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.text = json.dumps(vertex_body)
        mock_resp.json.return_value = vertex_body
        mock_resp.headers = httpx.Headers({"content-type": "application/json"})
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    @patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
    def test_routing_reaches_sk_handler(self, mock_post):
        mock_post.return_value = self._make_mock_httpx_response()

        response = litellm.completion(
            model=PROVIDER_MODEL,
            messages=[{"role": "user", "content": "hello"}],
            api_key=API_KEY,
            api_base=API_BASE,
            vertex_project=PROJECT,
            vertex_location=LOCATION,
        )

        assert mock_post.called, "HTTP POST was not called"

        # Verify the URL used
        call_kwargs = mock_post.call_args
        called_url = call_kwargs.kwargs.get("url") or (
            call_kwargs.args[0] if call_kwargs.args else None
        )
        assert called_url is not None
        assert API_BASE in called_url
        assert f"key={API_KEY}" in called_url
        assert PROJECT in called_url
        assert LOCATION in called_url

        # Verify spillover header was sent
        called_headers = call_kwargs.kwargs.get("headers", {})
        assert called_headers.get(SK_SPILLOVER_HEADER) == SK_SPILLOVER_VALUE

        # Verify response is OpenAI-format
        assert response.choices[0].message.content == "Hi there!"

    @patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
    def test_no_bearer_token_in_request(self, mock_post):
        mock_post.return_value = self._make_mock_httpx_response()

        litellm.completion(
            model=PROVIDER_MODEL,
            messages=[{"role": "user", "content": "hello"}],
            api_key=API_KEY,
            api_base=API_BASE,
            vertex_project=PROJECT,
            vertex_location=LOCATION,
        )

        call_kwargs = mock_post.call_args
        called_headers = call_kwargs.kwargs.get("headers", {})
        assert "Authorization" not in called_headers, (
            "Bearer token should NOT be sent for sk_vertex_ai"
        )
