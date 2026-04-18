"""Tests for the provider layer — factory, base interface, typed errors."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.providers import get_provider
from src.providers.base import (
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
    """OllamaProvider now owns a persistent httpx.Client — we patch the
    client's methods on the instance rather than the module-level httpx
    functions (which the provider no longer calls)."""

    def _build(self, **overrides):
        """Return (provider, mock_client) with the client instance patched."""
        from src.providers.ollama_provider import OllamaProvider
        p = OllamaProvider()
        p._client = MagicMock()
        for attr, value in overrides.items():
            setattr(p._client, attr, value)
        return p, p._client

    def test_list_models_success(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "llama3"}, {"name": "mistral"}]}
        p, client = self._build(get=MagicMock(return_value=mock_resp))
        models = p.list_models()
        assert "llama3" in models
        assert "mistral" in models

    def test_list_models_connection_error(self):
        p, client = self._build(get=MagicMock(side_effect=httpx.ConnectError("connection refused")))
        with pytest.raises(ProviderConnectionError, match="Cannot reach Ollama"):
            p.list_models()

    def test_list_models_timeout(self):
        p, _ = self._build(get=MagicMock(side_effect=httpx.ConnectTimeout("timed out")))
        with pytest.raises(ProviderConnectionError):
            p.list_models()

    def test_list_models_http_status_error(self):
        resp = httpx.Response(500, request=httpx.Request("GET", "http://localhost:11434/api/tags"))
        p, _ = self._build(get=MagicMock(
            side_effect=httpx.HTTPStatusError("server error", request=resp.request, response=resp)
        ))
        with pytest.raises(ProviderError, match="HTTP 500"):
            p.list_models()

    def test_list_models_empty_is_valid(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"models": []}
        p, _ = self._build(get=MagicMock(return_value=mock_resp))
        assert p.list_models() == []

    def test_is_available_returns_false_on_failure(self):
        p, _ = self._build(get=MagicMock(side_effect=httpx.ConnectError("refused")))
        assert p.is_available() is False

    def test_generate_connection_error(self):
        p, _ = self._build(post=MagicMock(side_effect=httpx.ConnectError("refused")))
        with pytest.raises(ProviderConnectionError):
            p.generate("hello", model="llama3")

    def test_generate_error_key_in_response(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"error": "model not found"}
        p, _ = self._build(post=MagicMock(return_value=mock_resp))
        with pytest.raises(ProviderError, match="model not found"):
            p.generate("hello", model="nonexistent")

    def test_persistent_client_is_reused(self):
        """Verify the provider keeps a single httpx.Client across calls."""
        from src.providers.ollama_provider import OllamaProvider
        p = OllamaProvider()
        first = p._client
        # Simulate another call — the client should still be the same instance.
        assert p._client is first
        p.close()


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


# ======================================================================
# Modernisation / hardening coverage
# ======================================================================

class TestOpenAITokenParam:
    """N3: reasoning models (o-series, gpt-5*) require max_completion_tokens."""

    def _stub_response(self):
        choice = MagicMock()
        choice.message.content = "ok"
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def _run(self, model: str):
        from src.providers.openai_provider import OpenAIProvider

        with patch("src.providers.openai_provider.OpenAIProvider._client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = self._stub_response()
            p = OpenAIProvider(api_key="sk-test")
            p.generate("hello", model=model)
            return mock_client.return_value.chat.completions.create.call_args.kwargs

    def test_classic_chat_uses_max_tokens(self):
        kwargs = self._run("gpt-4.1")
        assert "max_tokens" in kwargs and "max_completion_tokens" not in kwargs

    def test_o_series_uses_max_completion_tokens(self):
        kwargs = self._run("o3-mini")
        assert "max_completion_tokens" in kwargs and "max_tokens" not in kwargs

    def test_gpt5_uses_max_completion_tokens(self):
        kwargs = self._run("gpt-5-turbo")
        assert "max_completion_tokens" in kwargs and "max_tokens" not in kwargs

    def test_omni_model_keeps_max_tokens(self):
        # Edge case: model names starting with "omni" are classic chat models,
        # not reasoning models — the branch must not match them.
        kwargs = self._run("omni-turbo")
        assert "max_tokens" in kwargs


class TestAnthropicPromptCaching:
    """P5: system prompt must ship as a cache_control'd text block when present."""

    def _stub_response(self, text="hi"):
        block = MagicMock()
        block.text = text
        resp = MagicMock()
        resp.content = [block]
        return resp

    def test_system_is_wrapped_with_cache_control(self):
        from src.providers.anthropic_provider import AnthropicProvider

        with patch("src.providers.anthropic_provider.AnthropicProvider._client") as mock_client:
            mock_client.return_value.messages.create.return_value = self._stub_response()
            p = AnthropicProvider(api_key="test-key")
            p.generate("Describe the scene.", model="claude-sonnet-4-6", system="be precise")

            kwargs = mock_client.return_value.messages.create.call_args.kwargs
            assert isinstance(kwargs["system"], list)
            assert kwargs["system"][0]["type"] == "text"
            assert kwargs["system"][0]["text"] == "be precise"
            assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
            assert kwargs["max_tokens"] == 2048

    def test_no_system_means_no_system_param(self):
        from src.providers.anthropic_provider import AnthropicProvider

        with patch("src.providers.anthropic_provider.AnthropicProvider._client") as mock_client:
            mock_client.return_value.messages.create.return_value = self._stub_response()
            p = AnthropicProvider(api_key="test-key")
            p.generate("Describe the scene.", model="claude-sonnet-4-6")
            kwargs = mock_client.return_value.messages.create.call_args.kwargs
            assert "system" not in kwargs


class TestAnthropicImagePayload:
    def test_generate_with_image_sends_base64_block(self):
        import numpy as np

        from src.providers.anthropic_provider import AnthropicProvider

        block = MagicMock()
        block.text = "ok"
        resp = MagicMock()
        resp.content = [block]

        with patch("src.providers.anthropic_provider.AnthropicProvider._client") as mock_client:
            mock_client.return_value.messages.create.return_value = resp
            p = AnthropicProvider(api_key="test-key")
            img = np.zeros((8, 8, 3), dtype=np.uint8)
            p.generate("describe", model="claude-sonnet-4-6", images=[img])

            kwargs = mock_client.return_value.messages.create.call_args.kwargs
            content = kwargs["messages"][0]["content"]
            # first block should be the image, second the text prompt
            assert content[0]["type"] == "image"
            assert content[0]["source"]["type"] == "base64"
            assert content[0]["source"]["media_type"] == "image/jpeg"
            assert content[-1]["type"] == "text"


class TestGeminiGenerateImagePath:
    """N2: images must become ``Part.from_bytes``; system must reach GenerateContentConfig."""

    def test_generate_with_image_uses_part_from_bytes(self):
        import numpy as np

        from src.providers.gemini_provider import GeminiProvider

        resp = MagicMock()
        resp.text = "ok"
        with patch("src.providers.gemini_provider.GeminiProvider._client") as mock_client, \
             patch("google.genai.types.Part.from_bytes") as mock_part, \
             patch("google.genai.types.GenerateContentConfig") as mock_config:
            mock_client.return_value.models.generate_content.return_value = resp
            mock_part.return_value = "<image-part>"
            mock_config.return_value = "<config>"

            p = GeminiProvider(api_key="k")
            img = np.zeros((8, 8, 3), dtype=np.uint8)
            p.generate("describe", model="gemini-2.0-flash", images=[img], system="precise")

            mock_part.assert_called_once()
            args, kwargs = mock_part.call_args
            assert kwargs["mime_type"] == "image/jpeg"
            assert isinstance(kwargs["data"], (bytes, bytearray))

            mock_config.assert_called_once_with(system_instruction="precise")

            gen_kwargs = mock_client.return_value.models.generate_content.call_args.kwargs
            assert gen_kwargs["model"] == "gemini-2.0-flash"
            assert gen_kwargs["contents"][0] == "<image-part>"
            assert gen_kwargs["contents"][-1] == "describe"
            assert gen_kwargs["config"] == "<config>"


class TestImageEncodeCache:
    """P4: repeated encodes of the same frame hit the LRU cache."""

    def test_same_frame_encoded_once(self):
        import numpy as np

        from src.providers.base import LLMProvider, _cached_jpeg_encode

        _cached_jpeg_encode.cache_clear()
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)

        # Three encodes of the same frame — the LRU store should only run once.
        LLMProvider._encode_image_jpeg(img)
        LLMProvider._encode_image_jpeg(img)
        LLMProvider._encode_image_jpeg(img)

        info = _cached_jpeg_encode.cache_info()
        assert info.hits >= 2 and info.misses == 1

    def test_different_frame_misses(self):
        import numpy as np

        from src.providers.base import LLMProvider, _cached_jpeg_encode

        _cached_jpeg_encode.cache_clear()
        img1 = np.zeros((8, 8, 3), dtype=np.uint8)
        img2 = np.ones((8, 8, 3), dtype=np.uint8) * 255
        LLMProvider._encode_image_jpeg(img1)
        LLMProvider._encode_image_jpeg(img2)
        info = _cached_jpeg_encode.cache_info()
        assert info.misses == 2
