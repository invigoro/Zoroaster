import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from scripts.rank_v2 import tree_logprobs, tree_mask


def _tiny_model():
    """A random 2-layer Qwen2. Its query and key weights are scaled up so
    attention is sharp: with the default initialization it's nearly uniform,
    and wrong positions or masks barely change the scores."""
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    config._attn_implementation = "sdpa"
    model = Qwen2ForCausalLM(config).eval()
    with torch.no_grad():
        for layer in model.model.layers:
            layer.self_attn.q_proj.weight *= 40
            layer.self_attn.k_proj.weight *= 40
    return model


def _alone(model, base, continuation):
    with torch.no_grad():
        logp = torch.log_softmax(model(input_ids=torch.tensor([base + continuation])).logits[0].float(), dim=-1)
    return sum(logp[len(base) - 1 + j, t].item() for j, t in enumerate(continuation))


class TreeTest(unittest.TestCase):
    def test_mask_lets_nodes_see_the_prompt_and_their_ancestors(self):
        mask = tree_mask(2, [-1, 0, 0, -1]).int().tolist()  # nodes: a, a>b, a>c, d
        self.assertEqual(mask, [[1, 0, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0],
                                [1, 1, 1, 1, 0, 0],
                                [1, 1, 1, 0, 1, 0],
                                [1, 1, 0, 0, 0, 1]])

    def test_tree_scores_match_scoring_each_sequence_alone(self):
        model = _tiny_model()
        base = [5, 9, 12, 3, 7]
        continuations = [[1], [1, 2, 3], [1, 2, 4], [4, 5, 6, 7], [1, 3], [8]]  # shared prefixes, ragged
        got = tree_logprobs(model, base, continuations)
        want = [_alone(model, base, c) for c in continuations]
        for g, w in zip(got, want):
            self.assertAlmostEqual(g, w, places=4)
        self.assertEqual(len({round(w, 3) for w in want}), len(want))  # all differ, so a mix-up would show


if __name__ == "__main__":
    unittest.main()
