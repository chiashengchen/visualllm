"""First-clause early-flush text aggregator (TTFO optimization).

WHY: CosyVoice's first-chunk latency scales with the *input sentence length* -- it
prefills the whole sentence before emitting the first audio token. Measured: a 16-word
opening sentence costs ~3.0s to first audio vs ~1.7s for a short one. The LLM's long
first sentence is therefore the single biggest piece of TTFO (see docs / measure_report).

WHAT: flush the FIRST unit of each bot turn EARLY -- at the first clause boundary
(comma/semicolon/colon) once a minimum length is reached, or a hard max char count --
so the bot starts speaking on a short opening clause (~2.2s vs ~3.1s to first audio),
then revert to normal sentence aggregation for the rest of the turn. The remaining audio
streams in behind the first clause; CosyVoice runs faster than real-time, so it catches up.

TUNABLE (load-bearing): the first clause must produce enough audio to cover the NEXT
piece's synthesis startup, or the voice PAUSES between the opening clause and the rest --
especially under shared-GPU contention (the avatar renders the first clause's lips while
CosyVoice synthesizes the next piece). Too short a first clause = fast start but a gap;
too long = no gap but a slower start. Tune MIN/MAX until it starts fast AND flows.

Env knobs (read via os.getenv; .env is load_dotenv'd by pipeline.config at import):
  COSYVOICE_FIRST_PIECE=1            enable (default OFF -> plain SimpleTextAggregator)
  COSYVOICE_FIRST_PIECE_MIN_CHARS   min chars before a clause comma may trigger the flush (default 24)
  COSYVOICE_FIRST_PIECE_MAX_CHARS   force the flush at the next space past this many chars (default 60)

Only the FIRST piece per turn is affected; the flag resets on flush() (turn end),
reset(), and handle_interruption().
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from pipecat.utils.text.base_text_aggregator import Aggregation, AggregationType
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

_CLAUSE_PUNCT = ",;:"


class FirstClauseAggregator(SimpleTextAggregator):
    """SimpleTextAggregator that emits a short opening clause first, then normal sentences."""

    def __init__(self, *, min_chars: int = 24, max_chars: int = 60, **kwargs):
        super().__init__(**kwargs)
        self._min_chars = int(min_chars)
        self._max_chars = int(max_chars)
        self._first_done = False

    async def aggregate(self, text: str) -> AsyncIterator[Aggregation]:
        # Once the opening clause is out (or in TOKEN/WORD mode) behave exactly like the parent.
        if self._first_done or self._aggregation_type != AggregationType.SENTENCE:
            async for agg in super().aggregate(text):
                yield agg
            return

        for char in text:
            self._text += char

            # 1) A natural sentence actually ended before we hit a clause -> honor it (short opener).
            res = await self._check_sentence_with_lookahead(char)
            if res:
                self._first_done = True
                yield res
                continue

            if self._first_done:
                continue

            stripped = self._text.strip()

            # 2) Early clause flush: first comma/;/: past the minimum length.
            if len(stripped) >= self._min_chars and char in _CLAUSE_PUNCT:
                self._first_done = True
                self._text = ""
                self._needs_lookahead = False
                yield Aggregation(text=stripped.rstrip(_CLAUSE_PUNCT + " "), type="clause")
                continue

            # 3) Hard cap: force a flush at the next space so we never cut mid-word.
            if len(stripped) >= self._max_chars and char == " ":
                self._first_done = True
                self._text = ""
                self._needs_lookahead = False
                yield Aggregation(text=stripped, type="clause")
                continue

    async def flush(self) -> Aggregation | None:
        self._first_done = False  # next turn opens fresh (parent.flush may not call reset if empty)
        return await super().flush()

    async def handle_interruption(self):
        self._first_done = False
        await super().handle_interruption()

    async def reset(self):
        self._first_done = False
        await super().reset()
