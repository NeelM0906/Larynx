"""Scaffold tests for ConversationSession.

These exercise the event-queue shape + the stub _dispatch. Real handler
tests land with the implementation commit (task #6).
"""

from __future__ import annotations

import asyncio

import pytest

from larynx_gateway.services.conversation_service import (
    ConversationConfig,
    ConversationSession,
    LLMEvent,
    STTEvent,
    SessionState,
    TTSEvent,
    VADEvent,
    WSControlEvent,
)


def _session() -> ConversationSession:
    return ConversationSession(cfg=ConversationConfig(), session_id="test")


def test_initial_state_is_idle() -> None:
    s = _session()
    assert s.state is SessionState.IDLE
    assert s.session_id == "test"


@pytest.mark.asyncio
async def test_stop_exits_run_cleanly_with_no_events() -> None:
    s = _session()
    run_task = asyncio.create_task(s.run())
    await s.stop()
    await asyncio.wait_for(run_task, timeout=1.0)
    # No exception = clean exit.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event",
    [
        VADEvent(kind="speech_start", utterance_ordinal=1, session_ms=0),
        VADEvent(kind="speech_end", utterance_ordinal=1, session_ms=1000),
        STTEvent(kind="transcript_partial", utterance_ordinal=1, text="hi"),
        STTEvent(kind="transcript_final", utterance_ordinal=1, text="hi"),
        LLMEvent(kind="llm_token", utterance_ordinal=1, delta="hello"),
        LLMEvent(kind="llm_done", utterance_ordinal=1),
        TTSEvent(kind="tts_chunk", utterance_ordinal=1, pcm_s16le=b"\x00\x00", sample_rate=16000),
        TTSEvent(kind="tts_done", utterance_ordinal=1),
        WSControlEvent(payload={"type": "session.end"}),
    ],
)
async def test_dispatch_raises_assertion_on_every_event_type(event) -> None:
    """Scaffold stub: every event type must hit the AssertionError path.

    This is load-bearing during development — any pipeline that accidentally
    pushes events through the scaffold will fail loudly rather than silently
    drop work. Once handlers land (task #4) this test gets replaced by the
    real handler tests.
    """
    s = _session()
    run_task = asyncio.create_task(s.run())
    await s.enqueue(event)
    with pytest.raises(AssertionError, match="no handler wired"):
        await asyncio.wait_for(run_task, timeout=1.0)


@pytest.mark.asyncio
async def test_producers_never_mutate_state_through_enqueue() -> None:
    """Enqueue alone must not change observable session state.

    Guards the §1.1 invariant: producers push, consumer mutates. Until run()
    dispatches the event, nothing about the session should change.
    """
    s = _session()
    assert s.state is SessionState.IDLE
    await s.enqueue(VADEvent(kind="speech_start", utterance_ordinal=1, session_ms=0))
    # No consumer running — state must be unchanged.
    assert s.state is SessionState.IDLE
