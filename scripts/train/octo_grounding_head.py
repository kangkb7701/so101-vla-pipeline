"""Auxiliary visual grounding head for Octo fine-tuning.

This head is used only during training. It predicts normalized 2D visual
prompt labels from Octo's transformer readout tokens so the visual backbone
and transformer are explicitly penalized when they ignore the overlay.
"""

from typing import Dict

import flax.linen as nn
import jax
import jax.numpy as jnp

from octo.model.components.base import TokenGroup


class GroundingHead(nn.Module):
    """Predict normalized VP labels from a transformer readout.

    Output layout is:
      [point_x, point_y, box_x1, box_y1, box_x2, box_y2]
    Values are passed through sigmoid, so labels should be normalized to [0, 1].
    """

    readout_key: str = "readout_action"
    hidden_dim: int = 256
    output_dim: int = 6
    use_layer_norm: bool = True

    @nn.compact
    def __call__(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        train: bool = True,
    ) -> jax.Array:
        del train
        token_group = transformer_outputs[self.readout_key]
        if token_group.tokens.ndim != 4:
            raise ValueError(
                "Expected readout tokens with shape "
                "(batch, window, num_tokens, embedding), got "
                f"{token_group.tokens.shape}"
            )

        x = token_group.tokens.mean(axis=-2)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.output_dim)(x)
        return jnp.clip(nn.sigmoid(x), 1e-4, 1.0 - 1e-4)
