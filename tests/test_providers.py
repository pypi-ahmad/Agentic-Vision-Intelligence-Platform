"""Tests for the provider layer — factory, base interface, typed errors."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.providers import get_provider
from src.providers.base import (
    LLMProvider,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
)


class TestGetProvider:
    def test_ollama_provider(self):
        p = get_provider("Ollama")
        assert p is not None
        assert p.name == "Ollama"

    def test_openai_requires_key(self):
        with pytest.raises(ProviderAuthError, match="API key"):
            get_provider("OpenAI", api_key="")

    def test_gemini_requires_key(self):
        with pytest.raises(ProviderAuthError, match="API key"):
            get_provider("Gemini", api_key="")

    def test_anthropic_requires_key(self):
        with pytest.raises(ProviderAuthError, match="API key"):
            get_provider("Anthropic", api_key="")

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_provider("NoSuch")

    def test_openai_with_key(self):
        p = get_provider("OpenAI", api_key="sk-test")
        assert p.name == "OpenAI"

    def test_gemini_with_key(self):
        p = get_provider("Gemini", api_key="test-key")
        assert p.name == "Gemini"

    def test_anthropic_with_key(self):
        p = get_provider("Anthropic", api_key="test-key")
        assert p.name == "Anthropic"

    def test_provider_auth_error_is_provider_error(self):
        """ProviderAuthError should be catchable as ProviderError."""
        with pytest.raises(ProviderError):
            get_provider("OpenAI", api_key="")


# ======================================================================
# Ollama
# ======================================================================

class TestOllamaProvider:
    @patch("src.providers.ollama_provider.httpx.get")
    def test_list_models_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "llama3"}, {"name": "mistral"}]}
        mock_get.return_value = mock_resp

        from src.providers.ollama_provider import OllamaProvider
        p = OllamaProvider()
        models = p.list_models()
        assert "llama3" in models
        assert "mistral" in models

    @patch("src.providers.ollama_provider.httpx.get")
    def test_list_models_connection_error(self, mock_get):
        mock_get.side_effect = httpx.ConnectError("connection refused")

        from src.providers.ollama_provider import OllamaProvider
        p = OllamaProvider()
        with pytest.raises(ProviderConnectionError, match="Cannot reach Ollama"):
            p.list_models()

    @patch("src.providers.ollama_provider.httpx.get")
    def test_list_models_timeout(self, mock_get):
        mock_get.side_effect = httpx.ConnectTimeout("timed out")

        from src.providers.ollama_provider import OllamaProvider
        p = OllamaProvider()
        with pytest.raises(ProviderConnectionError):
            p.list_models()

    @patch("src.providers.ollama_provider.httpx.get")
    def test_list_models_http_status_error(self, mock_get):
        resp = httpx.Response(500, request=httpx.Request("GET", "http://localhost:11434/api/tags"))
        mock_get.side_effect = httpx.HTTPStatusError("server error", request=resp.request, response=resp)

        from src.providers.ollama_provider import OllamaProvider
        p = OllamaProvider()
        with pytest.raises(ProviderError, match="HTTP 500"):
            p.list_models()

    @patch("src.providers.ollama_provider.httpx.get")
    def test_list_models_empty_is_valid(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"models": []}
        mock_get.return_value = mock_resp

        from src.providers.ollama_provider import OllamaProvider
        p = OllamaProvider()
        assert p.list_models() == []

    @patch("src.providers.ollama_provider.httpx.get")
    def test_is_available_returns_false_on_failure(self, mock_get):
        mock_get.side_effect = httpx.ConnectError("refused")

        from src.providers.ollama_provider import OllamaProvider
        p = OllamaProvider()
        assert p.is_available() is False

    @patch("src.providers.ollama_provider.httpx.post")
    def test_generate_connection_error(self, mock_post):
        mock_post.side_effect = httpx.ConnectError("refused")

        from src.providers.ollama_provider import OllamaProvider
        p = OllamaProvider()
        with pytest.raises(ProviderConnectionError):
            p.generate("hello", model="llama3")

    @patch("src.providers.ollama_provider.httpx.post")
    def test_generate_error_key_in_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"error": "model not found"}
        mock_post.return_value = mock_resp

        from src.providers.ollama_provider import OllamaProvider
        p = OllamaProvider()
        with pytest.raises(ProviderError, match="model not found"):
            p.generate("hello", model="nonexistent")


# ======================================================================
# OpenAI
# ======================================================================

class TestOpenAIProvider:
    def test_list_models_filters_non_chat_models(self):
        class Model:
            def __init__(self, model_id: str):
                self.id = model_id

        fake_client = MagicMock()
        fake_client.models.list.return_value = [
            Model("gpt-4.1"),
            Model("o3-mini"),
            Model("text-embedding-3-large"),
            Model("whisper-1"),
            Model("omni-moderation-latest"),
        ]

        with patch("src.providers.openai_provider.OpenAIProvider._client", return_value=fake_client):
            from src.providers.openai_provider import OpenAIProvider

            provider = OpenAIProvider(api_key="sk-test")
            models = provider.list_models()

        assert "gpt-4.1" in models
        assert "o3-mini" in models
        assert "text-embedding-3-large" not in models
        assert "whisper-1" not in models
        assert "omni-moderation-latest" not in models

    def test_list_models_auth_error(self):
        import openai as _openai

        with patch("src.providers.openai_provider.OpenAIProvider._client") as mock_client:
            mock_client.return_value.models.list.side_effect = _openai.AuthenticationError(
                message="Invalid API key",
                response=MagicMock(status_code=401),
                body=None,
            )
            from src.providers.openai_provider import OpenAIProvider
            p = OpenAIProvider(api_key="bad-key")
            with pytest.raises(ProviderAuthError, match="Invalid OpenAI API key"):
                p.list_models()

    def test_list_models_rate_limit(self):
        import openai as _openai

        with patch("src.providers.openai_provider.OpenAIProvider._client") as mock_client:
            mock_client.return_value.models.list.side_effect = _openai.RateLimitError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429),
                body=None,
            )
            from src.providers.openai_provider import OpenAIProvider
            p = OpenAIProvider(api_key="sk-test")
            with pytest.raises(ProviderRateLimitError):
                p.list_models()

    def test_list_models_connection_error(self):
        import openai as _openai

        with patch("src.providers.openai_provider.OpenAIProvider._client") as mock_client:
            mock_client.return_value.models.list.side_effect = _openai.APIConnectionError(
                request=MagicMock(),
            )
            from src.providers.openai_provider import OpenAIProvider
            p = OpenAIProvider(api_key="sk-test")
            with pytest.raises(ProviderConnectionError):
                p.list_models()

    def test_generate_auth_error(self):
        import openai as _openai

        with patch("src.providers.openai_provider.OpenAIProvider._client") as mock_client:
            mock_client.return_value.chat.completions.create.side_effect = _openai.AuthenticationError(
                message="Invalid API key",
                response=MagicMock(status_code=401),
                body=None,
            )
            from src.providers.openai_provider import OpenAIProvider
            p = OpenAIProvider(api_key="bad-key")
            with pytest.raises(ProviderAuthError):
                p.generate("hello", model="gpt-4o")


# ======================================================================
# Gemini
# ======================================================================

class TestGeminiProvider:
    def test_list_models_auth_error(self):
        with patch("src.providers.gemini_provider.GeminiProvider._client") as mock_client:
            mock_client.return_value.models.list.side_effect = Exception("API key not valid")
            from src.providers.gemini_provider import GeminiProvider
            p = GeminiProvider(api_key="bad-key")
            with pytest.raises(ProviderAuthError, match="Invalid Gemini API key"):
                p.list_models()

    def test_list_models_rate_limit(self):
        with patch("src.providers.gemini_provider.GeminiProvider._client") as mock_client:
            mock_client.return_value.models.list.side_effect = Exception("429 quota exceeded")
            from src.providers.gemini_provider import GeminiProvider
            p = GeminiProvider(api_key="test-key")
            with pytest.raises(ProviderRateLimitError):
                p.list_models()

    def test_list_models_connection_error(self):
        with patch("src.providers.gemini_provider.GeminiProvider._client") as mock_client:
            mock_client.return_value.models.list.side_effect = Exception("connection timeout")
            from src.providers.gemini_provider import GeminiProvider
            p = GeminiProvider(api_key="test-key")
            with pytest.raises(ProviderConnectionError):
                p.list_models()

    def test_list_models_generic_error(self):
        with patch("src.providers.gemini_provider.GeminiProvider._client") as mock_client:
            mock_client.return_value.models.list.side_effect = Exception("something unexpected")
            from src.providers.gemini_provider import GeminiProvider
            p = GeminiProvider(api_key="test-key")
            with pytest.raises(ProviderError, match="something unexpected"):
                p.list_models()

    def test_list_models_success(self):
        class FakeModel:
            def __init__(self, name):
                self.name = name

        with patch("src.providers.gemini_provider.GeminiProvider._client") as mock_client:
            mock_client.return_value.models.list.return_value = [
                FakeModel("models/gemini-2.0-flash"),
                FakeModel("models/gemini-1.5-pro"),
                FakeModel("models/text-bison-001"),
            ]
            from src.providers.gemini_provider import GeminiProvider
            p = GeminiProvider(api_key="test-key")
            models = p.list_models()
            assert "gemini-2.0-flash" in models
            assert "gemini-1.5-pro" in models
            assert "text-bison-001" not in models  # filtered out (no "gemini")

    def test_generate_auth_error(self):
        with patch("src.providers.gemini_provider.GeminiProvider._client") as mock_client:
            mock_client.return_value.models.generate_content.side_effect = Exception("API key not valid")
            from src.providers.gemini_provider import GeminiProvider
            p = GeminiProvider(api_key="bad-key")
            with pytest.raises(ProviderAuthError):
                p.generate("hello", model="gemini-2.0-flash")


# ======================================================================
# Anthropic
# ======================================================================

class TestAnthropicProvider:
    def test_list_models_auth_error(self):
        import anthropic as _anthropic

        with patch("src.providers.anthropic_provider.AnthropicProvider._client") as mock_client:
            mock_client.return_value.models.list.side_effect = _anthropic.AuthenticationError(
                message="Invalid API key",
                response=MagicMock(status_code=401),
                body=None,
            )
            from src.providers.anthropic_provider import AnthropicProvider
            p = AnthropicProvider(api_key="bad-key")
            with pytest.raises(ProviderAuthError, match="Invalid Anthropic API key"):
                p.list_models()

    def test_list_models_rate_limit(self):
        import anthropic as _anthropic

        with patch("src.providers.anthropic_provider.AnthropicProvider._client") as mock_client:
            mock_client.return_value.models.list.side_effect = _anthropic.RateLimitError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429),
                body=None,
            )
            from src.providers.anthropic_provider import AnthropicProvider
            p = AnthropicProvider(api_key="test-key")
            with pytest.raises(ProviderRateLimitError):
                p.list_models()

    def test_list_models_connection_error(self):
        import anthropic as _anthropic

        with patch("src.providers.anthropic_provider.AnthropicProvider._client") as mock_client:
            mock_client.return_value.models.list.side_effect = _anthropic.APIConnectionError(
                request=MagicMock(),
            )
            from src.providers.anthropic_provider import AnthropicProvider
            p = AnthropicProvider(api_key="test-key")
            with pytest.raises(ProviderConnectionError):
                p.list_models()

    def test_list_models_success(self):
        class FakeModel:
            def __init__(self, model_id):
                self.id = model_id

        class FakeResp:
            data = [FakeModel("claude-sonnet-4-20250514"), FakeModel("claude-haiku-4-20250514")]

        with patch("src.providers.anthropic_provider.AnthropicProvider._client") as mock_client:
            mock_client.return_value.models.list.return_value = FakeResp()
            from src.providers.anthropic_provider import AnthropicProvider
            p = AnthropicProvider(api_key="test-key")
            models = p.list_models()
            assert "claude-sonnet-4-20250514" in models
            assert "claude-haiku-4-20250514" in models
