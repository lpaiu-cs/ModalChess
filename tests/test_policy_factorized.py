import torch

from modalchess.models.heads.policy_factorized import PolicyFactorizedHead, score_factorized_moves


def test_policy_factorized_head_output_shapes() -> None:
    head = PolicyFactorizedHead(d_model=32, use_pair_scorer=True)
    tokens = torch.randn(2, 64, 32)
    pooled = torch.randn(2, 32)
    outputs = head(tokens, pooled)
    assert outputs["src_logits"].shape == (2, 64)
    assert outputs["dst_logits"].shape == (2, 64)
    assert outputs["promo_logits"].shape == (2, 5)
    assert outputs["pair_logits"].shape == (2, 64, 64)


def test_policy_factorized_scoring() -> None:
    head = PolicyFactorizedHead(d_model=16)
    tokens = torch.randn(1, 64, 16)
    pooled = torch.randn(1, 16)
    outputs = head(tokens, pooled)
    scores = score_factorized_moves(
        {
            "src_logits": outputs["src_logits"][0],
            "dst_logits": outputs["dst_logits"][0],
            "promo_logits": outputs["promo_logits"][0],
        },
        [(12, 28, 0), (6, 21, 0)],
    )
    assert scores.shape == (2,)
