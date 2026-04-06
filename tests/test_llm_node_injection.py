"""
Unit tests for constructor-level LLM dependency injection in LLMNode.

Tests verify:
- LLM injection via ``llm=`` parameter at construction time.
- ``get_llm()`` returns the injected instance directly.
- ``llm=None`` falls back to ``default_llm()`` without raising.
- Two independently constructed nodes hold different LLM instances with no shared state.
- ``TranslatorEngNode.execute()`` and ``TranslatorUsrNode.execute()`` produce the correct
  output when a ``FakeListChatModel`` is injected — no API key, no network call.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# NOTE: FakeListChatModel lives in a private submodule (fake_chat_models) because
# langchain-core 0.3.x does not re-export it from the public ``fake`` module.
# Re-evaluate this import path when langchain-core adds a stable public re-export.
from langchain_core.language_models.fake_chat_models import FakeListChatModel

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from black_langcube.llm_modules.LLMNodes.LLMNode import LLMNode  # noqa: E402
from black_langcube.llm_modules.LLMNodes.subgraphs.translator_en_subgraph import (  # noqa: E402
    TranslatorEngNode,
)
from black_langcube.llm_modules.LLMNodes.subgraphs.translator_usr_subgraph import (  # noqa: E402
    TranslatorUsrNode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MinimalNode(LLMNode):
    """Minimal concrete LLMNode used only in unit tests."""

    def generate_messages(self):
        return [("human", "hello")]


def _fake_llm(responses=None):
    return FakeListChatModel(responses=responses or ["fake response"])


def _translator_en_state():
    return {
        "translation_input": "",
        "translation_output": "",
        "translation_tokens": {},
        "language": "",
        "question": "",
        "messages": [],
    }


def _translator_usr_state():
    return {
        "translation_input": "",
        "translation_output": "",
        "translation_tokens": {},
        "language": "",
        "messages": [],
    }


# ---------------------------------------------------------------------------
# Tests: LLMNode injection mechanics
# ---------------------------------------------------------------------------


class TestLLMNodeInjection(unittest.TestCase):
    """LLM injection mechanics on the base class."""

    def test_injected_llm_stored_as_instance_attribute(self):
        """Injected llm is stored and returned by get_llm()."""
        fake = _fake_llm()
        node = _MinimalNode({}, {}, llm=fake)
        self.assertIs(node.llm, fake)
        self.assertIs(node.get_llm(), fake)

    def test_get_llm_returns_stored_instance_not_factory_result(self):
        """get_llm() never calls a factory — it returns the stored attribute."""
        fake = _fake_llm(["response A"])
        node = _MinimalNode({}, {}, llm=fake)
        self.assertIs(node.get_llm(), fake)
        self.assertIs(node.get_llm(), fake)  # same object on repeated calls

    def test_two_nodes_with_different_llms_do_not_share_state(self):
        """Two nodes constructed independently hold separate LLM instances."""
        fake_a = _fake_llm(["A"])
        fake_b = _fake_llm(["B"])
        node_a = _MinimalNode({}, {}, llm=fake_a)
        node_b = _MinimalNode({}, {}, llm=fake_b)
        self.assertIsNot(node_a.llm, node_b.llm)
        self.assertIs(node_a.get_llm(), fake_a)
        self.assertIs(node_b.get_llm(), fake_b)

    def test_llm_none_falls_back_to_default_llm(self):
        """Passing llm=None triggers default_llm() without raising."""
        sentinel = MagicMock(name="default_llm_sentinel")
        with patch(
            "black_langcube.llm_modules.LLMNodes.LLMNode.default_llm",
            return_value=sentinel,
        ):
            node = _MinimalNode({}, {}, llm=None)
        self.assertIs(node.llm, sentinel)

    def test_omitting_llm_kwarg_falls_back_to_default_llm(self):
        """Omitting ``llm`` entirely triggers default_llm() without raising."""
        sentinel = MagicMock(name="default_llm_sentinel")
        with patch(
            "black_langcube.llm_modules.LLMNodes.LLMNode.default_llm",
            return_value=sentinel,
        ):
            node = _MinimalNode({}, {})
        self.assertIs(node.llm, sentinel)


# ---------------------------------------------------------------------------
# Tests: TranslatorEngNode with FakeListChatModel
# ---------------------------------------------------------------------------


class TestTranslatorEngNodeWithFakeLLM(unittest.IsolatedAsyncioTestCase):
    """TranslatorEngNode.execute() produces expected output with a fake LLM."""

    async def test_execute_returns_fake_translation(self):
        state = _translator_en_state()
        fake = _fake_llm(["Hola mundo"])
        node = TranslatorEngNode(state, {}, llm=fake)
        extra = {"translation_input": "Hello world", "language": "Spanish"}

        result = await node.execute(extra_input=extra)

        self.assertEqual(result["translation_output"], "Hola mundo")
        self.assertEqual(result["language"], "Spanish")
        self.assertIn("translation_tokens", result)

    async def test_execute_uses_injected_llm_not_factory(self):
        """No factory function is called during execute()."""
        state = _translator_en_state()
        fake = _fake_llm(["Bonjour le monde"])
        node = TranslatorEngNode(state, {}, llm=fake)
        extra = {"translation_input": "Hello world", "language": "French"}

        with patch(
            "black_langcube.llm_modules.LLMNodes.LLMNode.default_llm"
        ) as mock_factory:
            result = await node.execute(extra_input=extra)

        mock_factory.assert_not_called()
        self.assertEqual(result["translation_output"], "Bonjour le monde")

    def test_injected_llm_stored_on_translator_eng_node(self):
        fake = _fake_llm()
        node = TranslatorEngNode(_translator_en_state(), {}, llm=fake)
        self.assertIs(node.llm, fake)
        self.assertIs(node.get_llm(), fake)


# ---------------------------------------------------------------------------
# Tests: TranslatorUsrNode with FakeListChatModel
# ---------------------------------------------------------------------------


class TestTranslatorUsrNodeWithFakeLLM(unittest.IsolatedAsyncioTestCase):
    """TranslatorUsrNode.execute() produces expected output with a fake LLM."""

    async def test_execute_returns_fake_translation(self):
        state = _translator_usr_state()
        fake = _fake_llm(["Hola usuario"])
        node = TranslatorUsrNode(state, {}, llm=fake)
        extra = {"translation_input": "Hello user", "language": "Spanish"}

        result = await node.execute(extra_input=extra)

        self.assertEqual(result["translation_output"], "Hola usuario")
        self.assertEqual(result["language"], "Spanish")
        self.assertIn("translation_tokens", result)

    async def test_execute_uses_injected_llm_not_factory(self):
        """No factory function is called during execute()."""
        state = _translator_usr_state()
        fake = _fake_llm(["Hej bruger"])
        node = TranslatorUsrNode(state, {}, llm=fake)
        extra = {"translation_input": "Hello user", "language": "Danish"}

        with patch(
            "black_langcube.llm_modules.LLMNodes.LLMNode.default_llm"
        ) as mock_factory:
            result = await node.execute(extra_input=extra)

        mock_factory.assert_not_called()
        self.assertEqual(result["translation_output"], "Hej bruger")

    def test_injected_llm_stored_on_translator_usr_node(self):
        fake = _fake_llm()
        node = TranslatorUsrNode(_translator_usr_state(), {}, llm=fake)
        self.assertIs(node.llm, fake)
        self.assertIs(node.get_llm(), fake)


# ---------------------------------------------------------------------------
# Tests: singleton isolation between two independent nodes
# ---------------------------------------------------------------------------


class TestLLMSingletonIsolation(unittest.TestCase):
    """Two independently constructed node instances do not share LLM state."""

    def test_different_llm_instances_no_shared_state(self):
        """Injecting different LLMs into two nodes keeps them fully independent."""
        fake_en = _fake_llm(["English"])
        fake_es = _fake_llm(["Español"])

        node_en = TranslatorEngNode(_translator_en_state(), {}, llm=fake_en)
        node_es = TranslatorEngNode(_translator_en_state(), {}, llm=fake_es)

        self.assertIsNot(node_en.llm, node_es.llm)
        self.assertIs(node_en.get_llm(), fake_en)
        self.assertIs(node_es.get_llm(), fake_es)

    def test_eng_and_usr_nodes_independent(self):
        """TranslatorEngNode and TranslatorUsrNode hold separate LLM instances."""
        fake_eng = _fake_llm(["eng"])
        fake_usr = _fake_llm(["usr"])

        eng_node = TranslatorEngNode(_translator_en_state(), {}, llm=fake_eng)
        usr_node = TranslatorUsrNode(_translator_usr_state(), {}, llm=fake_usr)

        self.assertIsNot(eng_node.llm, usr_node.llm)
        self.assertIs(eng_node.get_llm(), fake_eng)
        self.assertIs(usr_node.get_llm(), fake_usr)


if __name__ == "__main__":
    unittest.main()
