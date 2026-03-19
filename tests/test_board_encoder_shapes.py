import torch

from modalchess.models.board_encoder import BoardEncoder


def test_board_encoder_output_shapes() -> None:
    model = BoardEncoder(
        history_length=2,
        input_channels=18,
        d_model=64,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2,
        dropout=0.0,
        use_relation_bias=True,
    )
    board_planes = torch.randn(3, 2, 18, 8, 8)
    extra_tokens = torch.randn(3, 2, 64)
    outputs = model(board_planes, extra_tokens=extra_tokens)
    assert outputs["tokens"].shape == (3, 64, 64)
    assert outputs["meta_tokens"].shape == (3, 2, 64)
    assert outputs["context_tokens"].shape == (3, 66, 64)
    assert outputs["board_pooled"].shape == (3, 64)
    assert outputs["context_pooled"].shape == (3, 64)
    assert outputs["pooled"].shape == (3, 64)
