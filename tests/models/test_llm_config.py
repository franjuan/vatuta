"""Unit tests for LLMBackendConfig and RagConfig validation models."""

import unittest

from pydantic import ValidationError

from src.models.config import LLMBackendConfig, RagConfig


class TestLLMConfig(unittest.TestCase):
    """Test suite for LLMBackendConfig and RagConfig validation rules."""

    def test_llm_backend_config_valid(self) -> None:
        """Verify valid LLMBackendConfig instantiation."""
        conf = LLMBackendConfig(
            model="gemini/gemini-2.5-flash",
            temperature=0.5,
            max_tokens=1000,
            top_p=0.9,
            api_base="https://custom.endpoint.com",
            extra_kwargs={"stream": False},
        )
        self.assertEqual(conf.model, "gemini/gemini-2.5-flash")
        self.assertEqual(conf.temperature, 0.5)
        self.assertEqual(conf.max_tokens, 1000)
        self.assertEqual(conf.top_p, 0.9)
        self.assertEqual(conf.api_base, "https://custom.endpoint.com")
        self.assertEqual(conf.extra_kwargs, {"stream": False})

    def test_llm_backend_config_alias_model_id(self) -> None:
        """Verify model_id alias populates model for backwards compatibility."""
        conf = LLMBackendConfig.model_validate(
            {
                "model_id": "gemini/gemini-1.5-pro",
                "temperature": 0.2,
            }
        )
        self.assertEqual(conf.model, "gemini/gemini-1.5-pro")

    def test_llm_backend_config_invalid_model_format(self) -> None:
        """Verify model string must conform to provider/model_name."""
        with self.assertRaises(ValidationError):
            LLMBackendConfig(model="invalidmodelwithoutslash")

    def test_llm_backend_config_invalid_temperature(self) -> None:
        """Verify temperature bounds [0.0, 2.0]."""
        with self.assertRaises(ValidationError):
            LLMBackendConfig(model="gemini/gemini-2.5-flash", temperature=-0.1)
        with self.assertRaises(ValidationError):
            LLMBackendConfig(model="gemini/gemini-2.5-flash", temperature=2.5)

    def test_llm_backend_config_invalid_max_tokens(self) -> None:
        """Verify max_tokens must be strictly positive."""
        with self.assertRaises(ValidationError):
            LLMBackendConfig(model="gemini/gemini-2.5-flash", max_tokens=0)
        with self.assertRaises(ValidationError):
            LLMBackendConfig(model="gemini/gemini-2.5-flash", max_tokens=-50)

    def test_llm_backend_config_invalid_top_p(self) -> None:
        """Verify top_p bounds [0.0, 1.0]."""
        with self.assertRaises(ValidationError):
            LLMBackendConfig(model="gemini/gemini-2.5-flash", top_p=1.5)

    def test_rag_config_valid(self) -> None:
        """Verify valid RagConfig with router and generator resolving to llm_backends."""
        conf = RagConfig(
            llm_backends={
                "gemini_flash": LLMBackendConfig(model="gemini/gemini-2.5-flash"),
                "gemini_pro": LLMBackendConfig(model="gemini/gemini-1.5-pro"),
            },
            router_backend="gemini_flash",
            generator_backend="gemini_pro",
        )
        self.assertIn("gemini_flash", conf.llm_backends)
        self.assertEqual(conf.router_backend, "gemini_flash")
        self.assertEqual(conf.generator_backend, "gemini_pro")

    def test_rag_config_empty_backends(self) -> None:
        """Verify RagConfig rejects empty llm_backends dictionary."""
        with self.assertRaises(ValidationError):
            RagConfig(
                llm_backends={},
                router_backend="gemini_flash",
                generator_backend="gemini_pro",
            )

    def test_rag_config_unresolved_router_backend(self) -> None:
        """Verify RagConfig raises ValidationError when router_backend is not in llm_backends."""
        with self.assertRaises(ValidationError) as ctx:
            RagConfig(
                llm_backends={"gemini_flash": LLMBackendConfig(model="gemini/gemini-2.5-flash")},
                router_backend="non_existent_router",
                generator_backend="gemini_flash",
            )
        self.assertIn("router_backend", str(ctx.exception))

    def test_rag_config_unresolved_generator_backend(self) -> None:
        """Verify RagConfig raises ValidationError when generator_backend is not in llm_backends."""
        with self.assertRaises(ValidationError) as ctx:
            RagConfig(
                llm_backends={"gemini_flash": LLMBackendConfig(model="gemini/gemini-2.5-flash")},
                router_backend="gemini_flash",
                generator_backend="non_existent_generator",
            )
        self.assertIn("generator_backend", str(ctx.exception))
