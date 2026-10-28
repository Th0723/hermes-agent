"""A fallback whose window cannot hold the request is a dead end, not a fallback.

Measured on this host (2026-10-28): a session whose last request measured 255,591
prompt tokens fell back to ``copilot/gpt-4.1`` (128,000-token window) and the provider
rejected it with ``model_max_prompt_tokens_exceeded`` before any answer — the session
was then reported as "cannot compress further" even though the PRIMARY model had a
1,000,000-token window that could have held it.

Compression cannot rescue the switch: the parts of the request that a plugin context
engine cannot rewrite (the tool schemas, the assistant tool-call arguments) are part of
every request, so the wire size stays above the smaller window no matter how hard the
compressor squeezes the message text.
"""
from types import SimpleNamespace
from unittest.mock import patch

from agent import chat_completion_helpers as cch


def _agent(*, measured: int = 0, real: int = 0) -> SimpleNamespace:
    agent = SimpleNamespace()
    agent.model = "MiniMax-M3.1-Flash"
    agent.provider = "minimax-cn"
    agent.base_url = "https://api.minimaxi.com/anthropic"
    agent._provider_fallback_active = False
    agent.context_compressor = SimpleNamespace(
        last_prompt_tokens=measured, last_real_prompt_tokens=real,
    )
    return agent


def test_candidate_window_smaller_than_the_measured_request_is_skipped():
    """The 14:12 shape: 255,591-token request, 128,000-token fallback window."""
    agent = _agent(measured=255_591)
    fb = {"provider": "copilot", "model": "gpt-4.1", "base_url": "https://api.githubcopilot.com"}
    with patch.object(cch, "_fallback_candidate_window", return_value=128_000):
        assert cch._fallback_window_cannot_hold_session(agent, fb, "copilot", "gpt-4.1") is True


def test_candidate_window_that_fits_the_request_is_kept():
    agent = _agent(measured=90_000)
    fb = {"provider": "copilot", "model": "gpt-4.1", "base_url": "https://api.githubcopilot.com"}
    with patch.object(cch, "_fallback_candidate_window", return_value=128_000):
        assert cch._fallback_window_cannot_hold_session(agent, fb, "copilot", "gpt-4.1") is False


def test_unknown_window_is_not_evidence():
    """A window we cannot resolve is not proof the candidate is too small."""
    agent = _agent(measured=255_591)
    fb = {"provider": "mystery", "model": "some-model"}
    with patch.object(cch, "_fallback_candidate_window", return_value=0):
        assert cch._fallback_window_cannot_hold_session(agent, fb, "mystery", "some-model") is False


def test_unmeasured_session_leaves_the_candidate_eligible():
    """Nothing measured yet -> no floor -> the candidate stays eligible."""
    agent = _agent(measured=0, real=0)
    fb = {"provider": "copilot", "model": "gpt-4.1"}
    with patch.object(cch, "_fallback_candidate_window", return_value=128_000):
        assert cch._fallback_window_cannot_hold_session(agent, fb, "copilot", "gpt-4.1") is False


def test_explicit_context_length_on_the_entry_wins_over_resolution():
    fb = {"provider": "copilot", "model": "gpt-4.1", "context_length": 1_000_000}
    with patch("agent.model_metadata.get_model_context_length", return_value=128_000):
        assert cch._fallback_candidate_window(fb, "copilot", "gpt-4.1") == 1_000_000


def test_candidate_window_is_resolved_with_the_candidates_own_credential():
    """The same slug carries a different limit per provider, so the candidate's own
    credential has to be used — otherwise the generic table's number is read instead."""
    fb = {"provider": "copilot", "model": "gpt-4.1", "base_url": "https://api.githubcopilot.com"}
    with patch("hermes_cli.fallback_config.resolve_entry_api_key", return_value="gho_candidate"), \
         patch("agent.model_metadata.get_model_context_length", return_value=128_000) as resolver:
        assert cch._fallback_candidate_window(fb, "copilot", "gpt-4.1") == 128_000
    assert resolver.call_args.kwargs.get("api_key") == "gho_candidate"
    assert resolver.call_args.kwargs.get("provider") == "copilot"


def test_should_skip_reports_the_window_guard():
    """The guard is wired into the candidate filter the fallback chain walks."""
    agent = _agent(measured=255_591)
    fb = {"provider": "copilot", "model": "gpt-4.1", "base_url": "https://api.githubcopilot.com"}
    fb_key = ("copilot", "gpt-4.1", "https://api.githubcopilot.com")
    with patch.object(cch, "_fallback_candidate_window", return_value=128_000), \
         patch.object(cch, "_candidate_pool_exhausted", return_value=False), \
         patch.object(cch, "_fallback_entry_unavailable_without_network", return_value=None):
        assert cch._should_skip_fallback_candidate(agent, fb, fb_key, "copilot", "gpt-4.1", set()) is True
