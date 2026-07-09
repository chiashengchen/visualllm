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
  COSYVOICE_FIRST_PIECE_ZH          opt-in zh path (default 0=OFF; needs COSYVOICE_FIRST_PIECE=1 too).
                                    WHY: the en rules above are a near-no-op for Chinese -- zh
                                    clauses end at the FULL-WIDTH comma U+FF0C and have no spaces,
                                    so the ASCII comma/space triggers never fire and a long zh
                                    opener still prefills whole (TTS first-chunk ~3.0s vs ~1.5s).
                                    The zh path flushes the first piece at a full-width ，；：
                                    ONLY -- never at a char cap, because a cap can cut mid-word
                                    (the rejected 天氣預|報 splitter); a comma boundary cannot.
                                    No comma in the sentence = identical to today (whole sentence).
  COSYVOICE_FIRST_PIECE_ZH_MIN_CHARS  min CJK chars before a ， may flush (default 5). WHY: the
                                    opening piece's audio must cover the NEXT piece's synthesis
                                    startup or the voice pauses between clauses (same mechanics as
                                    the en MIN); zh speaks ~4-5 chars/s so 5+ chars ~= 1.2s+ audio,
                                    and steady's lead cushion adds slack. A ， earlier than the min
                                    is skipped -- the flush waits for the NEXT clause boundary.

Only the FIRST piece per turn is affected; the flag resets on flush() (turn end),
reset(), and handle_interruption().
"""
from __future__ import annotations

import os
import random
import re
from collections.abc import AsyncIterator

from pipecat.utils.text.base_text_aggregator import Aggregation, AggregationType
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

_CLAUSE_PUNCT = ",;:"

# Filler ("thinking") words -- a natural utterance the bot speaks the instant a turn starts
# so it begins talking ~1s in (a tiny "嗯，" synthesizes in ~0.4s) while CosyVoice prefills the
# real first sentence behind it (~2s). Masks the TTS first-chunk wait exactly like a human "umm".
# Each ends on a SOFT trailing comma so it flows into the real reply (and gives the zh normalizer
# clause prosody). A large pool + no-immediate-repeat keeps it from sounding canned. Traditional
# zh (LANGUAGE=zh is zh-TW). Chain FILLER_WORDS_COUNT of them to bridge the gap on long openers.
# These are deliberately ~1–1.5s of speech each (not one-syllable "嗯，"): in steady mode the
# pump holds the voice until MUSETALK_LEAD_FRAMES frames render (~1s of audio at 14fps), so a
# TOO-SHORT first piece can't fill that cushion and the hold BALLOONS (measured: a 0.3s "嗯，"
# opener pushed the hold 0.5s->1.7s). A ~1.2s thinking phrase fills the cushion AND opens fast.
_ZH_FILLERS = [
    "嗯，讓我想一下喔，", "這個問題嘛，讓我想想，", "嗯，我來說說看，", "好的，關於這個呢，",
    "唔，讓我思考一下喔，", "嗯，這個部分呢，", "欸，說到這個問題，", "嗯，這樣子的話呢，",
    "好的，這個問題嘛，", "嗯，關於這一點呢，",
]
_EN_FILLERS = [
    "Well, let me think about that, ", "Hmm, that's a good question, ", "Okay, let me see now, ",
    "Right, so about that, ", "Let me think for a moment, ", "Ah, good question, so, ",
    "Okay, so on that, ", "Hmm, let me put it this way, ",
]

# zh clause boundaries: full-width comma/semicolon/colon. Deliberately NOT the enumeration
# comma 、 (joins list items too tightly to pause at) and NO char-cap fallback (cuts mid-word).
_ZH_CLAUSE_PUNCT = "，；："

# Same CJK-detection convention as the cosyvoice repo's tts_engine.py::is_cjk (CJK unified +
# compatibility ideographs + kana). Used both to gate the zh path to actually-Chinese text and
# to count "spoken" chars for the min guard (ASCII like "AI" is excluded -> conservative).
_CJK = re.compile(r"[㐀-鿿豈-﫿぀-ヿ]")


class FirstClauseAggregator(SimpleTextAggregator):
    """SimpleTextAggregator that emits a short opening clause first, then normal sentences."""

    def __init__(self, *, min_chars: int = 24, max_chars: int = 60, **kwargs):
        super().__init__(**kwargs)
        self._min_chars = int(min_chars)
        self._max_chars = int(max_chars)
        self._first_done = False
        # zh path knobs, read here (not plumbed through cosyvoice_tts.py) so the en call
        # site stays byte-identical; same truthy convention as the FIRST_PIECE gate itself.
        self._zh_enabled = os.getenv("COSYVOICE_FIRST_PIECE_ZH", "0").lower() in ("1", "true", "yes", "on")
        self._zh_min_chars = int(os.getenv("COSYVOICE_FIRST_PIECE_ZH_MIN_CHARS", "5") or "5")

        # Filler ("thinking") words -- see the pool comments above. OFF by default; when on, the
        # turn opens on FILLER_WORDS_COUNT canned fillers (fast to synth) before the real reply,
        # so TTFO is the filler's ~0.4s synth, not the real sentence's ~2s prefill.
        self._filler_enabled = os.getenv("FILLER_WORDS", "0").lower() in ("1", "true", "yes", "on")
        self._filler_count = max(1, int(os.getenv("FILLER_WORDS_COUNT", "2") or "2"))
        self._filler_pool = _ZH_FILLERS if os.getenv("LANGUAGE", "en").lower().startswith("zh") else _EN_FILLERS
        self._filler_emitted = False   # per-turn latch (reset on flush/reset/interruption)
        self._last_filler: str | None = None   # avoid repeating the previous turn's opener

    def _pick_fillers(self) -> list[str]:
        """FILLER_WORDS_COUNT distinct fillers; the opener differs from last turn's (no obvious repeat)."""
        pool = self._filler_pool
        n = min(self._filler_count, len(pool))
        first = random.choice([f for f in pool if f != self._last_filler] or pool)
        rest = random.sample([f for f in pool if f != first], max(0, n - 1))
        self._last_filler = first
        return [first, *rest]

    async def aggregate(self, text: str) -> AsyncIterator[Aggregation]:
        # Filler("thinking") words lead the turn -- emitted ONCE, before any other path, so the
        # bot starts speaking on the fast-to-synth filler instead of the slow real first sentence.
        if self._filler_enabled and not self._filler_emitted:
            self._filler_emitted = True
            for f in self._pick_fillers():
                yield Aggregation(text=f, type="clause")

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

            # 2a) zh early clause flush (opt-in): a full-width ，；： once enough CJK chars
            # are buffered to cover the next piece's synthesis (see the ZH_MIN_CHARS why).
            # Comma-only by design -- no cap fallback, so a comma-less zh sentence flows
            # through to the normal whole-sentence path unchanged.
            if (
                self._zh_enabled
                and char in _ZH_CLAUSE_PUNCT
                and len(_CJK.findall(stripped)) >= self._zh_min_chars
            ):
                self._first_done = True
                self._text = ""
                self._needs_lookahead = False
                # Keep the trailing ， -- CosyVoice's zh text-normalizer uses it for clause
                # prosody (unlike the en path, which strips its ASCII comma).
                yield Aggregation(text=stripped, type="clause")
                continue

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
        self._filler_emitted = False
        return await super().flush()

    async def handle_interruption(self):
        self._first_done = False
        self._filler_emitted = False
        await super().handle_interruption()

    async def reset(self):
        self._first_done = False
        self._filler_emitted = False
        await super().reset()
