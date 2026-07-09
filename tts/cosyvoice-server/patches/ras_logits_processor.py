"""Repetition-Aware Sampling (RAS) as a vLLM V1 logits processor.

Root cause this fixes: CosyVoice's native PyTorch decode samples with `ras_sampling`
(cosyvoice/utils/common.py) -- nucleus (top_p=0.8/top_k=25), then if the picked speech
token already appears in the last `win_size` decoded tokens it bans that token and
resamples. That windowed repeat guard is what stops the autoregressive speech-token LLM
from looping on the silence token. The vLLM decode path does its OWN sampling and never
applied RAS, so Chinese (inference_zero_shot) intermittently fell into the loop -> a ~4s
sentence stretched to ~12s of dead silence (heard as broken/halting speech, and the avatar
kept animating through the silence). vLLM's built-in repetition/frequency penalties can't
substitute: they build a prompt-token bincount, but CosyVoice feeds `prompt_embeds` (no
prompt token ids) -> a CUDA out-of-bounds device-side assert that kills the engine.

This reinstates RAS *inside* vLLM's sampler. A per-request logits processor bans any speech
token seen in the last COSYVOICE_RAS_WIN (default 10 = RAS's win_size) OUTPUT tokens, so a
recently emitted token -- the loop token -- can't be re-selected. Same anti-loop effect as
native RAS (which bans on rep_num >= win_size*tau_r = 10*0.1 = 1, i.e. any recurrence in the
window), using OUTPUT tokens only, so it is safe with the prompt_embeds input. Pair with
top_p=0.8 in SamplingParams (llm.py) to match RAS's nucleus. Set COSYVOICE_RAS_WIN=0 to
disable. Registered on the engine via EngineArgs(logits_processors=[...]) in cli/model.py.
"""
import os

import torch
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor


class RasLogitsProcessor(AdapterLogitsProcessor):
    """vLLM V1 adapter: reinstates CosyVoice's repetition-aware anti-loop guard."""

    def is_argmax_invariant(self) -> bool:
        # Banning recently emitted tokens can change which token is the argmax.
        return False

    def new_req_logits_processor(self, params):
        win = int(os.getenv("COSYVOICE_RAS_WIN", "10"))
        if win <= 0:
            return None

        def ras(output_ids, logits):
            # output_ids: this request's decoded speech tokens so far (a live reference).
            # logits: 1-D scores over the speech-token vocab for the next step.
            if output_ids:
                recent = output_ids[-win:]
                idx = torch.as_tensor(
                    sorted(set(recent)), device=logits.device, dtype=torch.long
                )
                logits.index_fill_(0, idx, float("-inf"))
            return logits

        return ras
