"""Custom AutoGen termination for the round-robin consensus debate.

Round-robin already takes strict turns (each agent speaks once per round, in
order). The stock `TextMentionTermination("consensus")` ends the debate the
instant *one* message mentions the token — even mid-stream or inside prose — so
later agents (notably the Critic, which speaks last) get skipped.

`DebateTermination` fixes that by only ending on a *full round* of agreement, and
adds two failure paths (round limit, stuck loop) that hand the closing word to
the Critic. The stop reason is carried in the `StopMessage.content` and parsed by
the caller (`backend/main.py`).
"""
from difflib import SequenceMatcher
from typing import Sequence

from autogen_agentchat.base import TerminationCondition, TerminatedException
from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    StopMessage,
    TextMessage,
)

# Stop-reason tokens (also the StopMessage content). Kept distinct so the caller
# can decide between the success path and the Critic's closing statement.
REASON_CONSENSUS = "unanimous-consensus"
REASON_MAX_ROUNDS = "max-rounds"
REASON_STUCK_LOOP = "stuck-loop"


def _letters_only(line: str) -> str:
    return "".join(c for c in line.lower() if c.isalpha())


def signals_consensus(text: str, token: str = "consensus") -> bool:
    """True only if some line, reduced to letters, equals the token.

    Matches the "reply with the token on its own line, nothing else" rule, so an
    agent mentioning the word in prose (e.g. "the consensus emerging…") does not
    falsely end the debate.
    """
    if not text:
        return False
    token = token.lower()
    return any(_letters_only(line) == token for line in text.splitlines())


def strip_consensus(text: str, token: str = "consensus") -> str:
    """Drop standalone-token lines; return the remaining substantive text."""
    if not text:
        return ""
    token = token.lower()
    kept = [ln for ln in text.splitlines() if _letters_only(ln) != token]
    return "\n".join(kept).strip()


class DebateTermination(TerminationCondition):
    """End a round-robin debate on one of three conditions.

    - **Unanimous consensus**: `num_agents` consecutive complete agent messages
      each signal the token (round-robin guarantees this spans one of every
      agent — a full agreeing round).
    - **Max rounds**: `max_rounds` rounds elapse (1 round = `num_agents`
      messages) without agreement.
    - **Stuck loop**: every agent repeats itself (text similarity ≥ `sim`) for
      `loop_threshold` consecutive rounds.

    Only complete `TextMessage`s from real agents are counted; streaming chunks,
    tool events, and the seed/user message are ignored.
    """

    def __init__(
        self,
        num_agents: int,
        max_rounds: int | None,
        loop_threshold: int = 3,
        sim: float = 0.85,
        token: str = "consensus",
    ) -> None:
        self._num_agents = max(1, num_agents)
        # None ⇒ no round cap (unlimited); consensus / stuck-loop still terminate.
        self._max_rounds = None if max_rounds is None else max(1, max_rounds)
        self._loop_threshold = max(1, loop_threshold)
        self._sim = sim
        self._token = token
        self._terminated = False
        self._count = 0  # complete agent messages seen
        self._streak = 0  # consecutive consensus signals
        self._last: dict[str, str] = {}  # last substantive text per agent
        self._repeat: dict[str, int] = {}  # consecutive near-duplicate rounds per agent

    @property
    def terminated(self) -> bool:
        return self._terminated

    async def __call__(
        self, messages: Sequence[BaseAgentEvent | BaseChatMessage]
    ) -> StopMessage | None:
        if self._terminated:
            raise TerminatedException("Termination condition has already been reached")

        for message in messages:
            # Only weigh complete agent answers.
            if not isinstance(message, TextMessage) or message.source == "user":
                continue

            text = message.content or ""
            src = message.source
            self._count += 1
            rounds_done = self._count // self._num_agents

            # 1) Unanimous consensus — a full round of agreement.
            if signals_consensus(text, self._token):
                self._streak += 1
            else:
                self._streak = 0
            if self._streak >= self._num_agents:
                return self._stop(REASON_CONSENSUS)

            # 2) Stuck loop — every agent re-asserting itself for N rounds.
            prev = self._last.get(src)
            if prev is not None and SequenceMatcher(None, prev, text).ratio() >= self._sim:
                self._repeat[src] = self._repeat.get(src, 0) + 1
            else:
                self._repeat[src] = 0
            self._last[src] = text
            if (
                len(self._repeat) >= self._num_agents
                and min(self._repeat.values()) >= self._loop_threshold
            ):
                return self._stop(REASON_STUCK_LOOP)

            # 3) Round limit reached without agreement (skipped when unlimited).
            if self._max_rounds is not None and rounds_done >= self._max_rounds:
                return self._stop(REASON_MAX_ROUNDS)

        return None

    def _stop(self, reason: str) -> StopMessage:
        self._terminated = True
        return StopMessage(content=reason, source="DebateTermination")

    async def reset(self) -> None:
        self._terminated = False
        self._count = 0
        self._streak = 0
        self._last.clear()
        self._repeat.clear()
