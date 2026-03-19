import torch

from modalchess.models.heads.state_probe import StateProbeHead


def test_state_probe_output_shapes() -> None:
    head = StateProbeHead(d_model=48)
    tokens = torch.randn(4, 64, 48)
    pooled = torch.randn(4, 48)
    outputs = head(tokens, pooled)
    assert outputs["square_state_logits"].shape == (4, 13, 8, 8)
    assert outputs["side_to_move_logits"].shape == (4,)
    assert outputs["castling_logits"].shape == (4, 4)
    assert outputs["en_passant_logits"].shape == (4, 65)
    assert outputs["in_check_logits"].shape == (4,)
