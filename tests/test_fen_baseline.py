import torch

from modalchess.models.fen_baseline import FenPolicyBaselineModel


def test_fen_baseline_output_shapes() -> None:
    model = FenPolicyBaselineModel(
        vocab_size=32,
        max_length=96,
        d_model=64,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        legality_hidden_dim=32,
        concept_vocab=["check", "capture"],
    )
    fen_token_ids = torch.randint(0, 32, (3, 24))
    fen_attention_mask = torch.ones(3, 24, dtype=torch.bool)
    outputs = model(fen_token_ids=fen_token_ids, fen_attention_mask=fen_attention_mask)
    assert outputs["tokens"].shape == (3, 64, 64)
    assert outputs["square_state_logits"].shape == (3, 13, 8, 8)
    assert outputs["legality_logits"].shape == (3, 64, 64, 5)
    assert outputs["src_logits"].shape == (3, 64)
