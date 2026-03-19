import torch

from modalchess.data.fen_tokenizer import FenTokenizer
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
    meta_features = torch.tensor(
        [[0.0, 1.0, 0.0], [12.0, 32.0, 2.0], [50.0, 80.0, 4.0]],
        dtype=torch.float32,
    )
    outputs = model(
        fen_token_ids=fen_token_ids,
        fen_attention_mask=fen_attention_mask,
        meta_features=meta_features,
    )
    assert outputs["tokens"].shape == (3, 64, 64)
    assert outputs["meta_tokens"].shape == (3, 2, 64)
    assert outputs["board_pooled"].shape == (3, 64)
    assert outputs["context_pooled"].shape == (3, 64)
    assert outputs["square_state_logits"].shape == (3, 13, 8, 8)
    assert outputs["legality_logits"].shape == (3, 64, 64, 5)
    assert outputs["src_logits"].shape == (3, 64)


def test_fen_baseline_uses_meta_features() -> None:
    model = FenPolicyBaselineModel(
        vocab_size=32,
        max_length=96,
        d_model=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        legality_hidden_dim=16,
        concept_vocab=["check"],
    )
    fen_token_ids = torch.randint(0, 32, (1, 24))
    fen_attention_mask = torch.ones(1, 24, dtype=torch.bool)
    meta_a = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
    meta_b = torch.tensor([[80.0, 120.0, 3.0]], dtype=torch.float32)
    outputs_a = model(
        fen_token_ids=fen_token_ids,
        fen_attention_mask=fen_attention_mask,
        meta_features=meta_a,
    )
    outputs_b = model(
        fen_token_ids=fen_token_ids,
        fen_attention_mask=fen_attention_mask,
        meta_features=meta_b,
    )
    assert not torch.allclose(outputs_a["meta_tokens"], outputs_b["meta_tokens"])
    assert not torch.allclose(outputs_a["context_pooled"], outputs_b["context_pooled"])


def test_fen_tokenizer_has_no_unknowns_for_fixture_fens() -> None:
    tokenizer = FenTokenizer.default()
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
        "r3k2r/pppq1ppp/2npbn2/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w kq - 0 8",
    ]
    unk_id = tokenizer.unk_token_id
    for fen in fens:
        assert unk_id not in tokenizer.encode(fen)
