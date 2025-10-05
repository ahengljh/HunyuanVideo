from typing import Any, List, Tuple, Optional, Union, Dict
from einops import rearrange

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from .activation_layers import get_activation_layer
from .norm_layers import get_norm_layer
from .embed_layers import TimestepEmbedder, PatchEmbed, TextProjection
from .attenion import attention, parallel_attention, get_cu_seqlens
from .posemb_layers import apply_rotary_emb
from .mlp_layers import MLP, MLPEmbedder, FinalLayer
from .modulate_layers import ModulateDiT, modulate, apply_gate
from .token_refiner import SingleTokenRefiner

logger = logging.getLogger(__name__)


class MMDoubleStreamBlock(nn.Module):
    """
    A multimodal dit block with seperate modulation for
    text and image/video, see more details (SD3): https://arxiv.org/abs/2403.03206
                                     (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.img_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.img_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs
        )
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )

        self.img_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.img_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )

        self.txt_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.txt_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.txt_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs
        )
        self.txt_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )

        self.txt_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.txt_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )
        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: tuple = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec).chunk(6, dim=-1)
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(vec).chunk(6, dim=-1)

        # Prepare image for attention.
        img_modulated = self.img_norm1(img)
        img_modulated = modulate(
            img_modulated, shift=img_mod1_shift, scale=img_mod1_scale
        )
        img_qkv = self.img_attn_qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # Apply QK-Norm if needed
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk

        # Prepare txt for attention.
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(
            txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale
        )
        txt_qkv = self.txt_attn_qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # Apply QK-Norm if needed.
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        # Run actual attention.
        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)
        v = torch.cat((img_v, txt_v), dim=1)
        assert (
            cu_seqlens_q.shape[0] == 2 * img.shape[0] + 1
        ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, img.shape[0]:{img.shape[0]}"
        
        # attention computation start
        if not self.hybrid_seq_parallel_attn:
            attn = attention(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=img_k.shape[0],
            )
        else:
            attn = parallel_attention(
                self.hybrid_seq_parallel_attn,
                q,
                k,
                v,
                img_q_len=img_q.shape[1],
                img_kv_len=img_k.shape[1],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv
            )
            
        # attention computation end

        img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]

        # Calculate the img bloks.
        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = img + apply_gate(
            self.img_mlp(
                modulate(
                    self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale
                )
            ),
            gate=img_mod2_gate,
        )

        # Calculate the txt bloks.
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(
                modulate(
                    self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale
                )
            ),
            gate=txt_mod2_gate,
        )

        return img, txt


class MMSingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    Also refer to (SD3): https://arxiv.org/abs/2403.03206
                  (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim ** -0.5

        # qkv and mlp_in
        self.linear1 = nn.Linear(
            hidden_size, hidden_size * 3 + mlp_hidden_dim, **factory_kwargs
        )
        # proj and mlp_out
        self.linear2 = nn.Linear(
            hidden_size + mlp_hidden_dim, hidden_size, **factory_kwargs
        )

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )

        self.pre_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.mlp_act = get_activation_layer(mlp_act_type)()
        self.modulation = ModulateDiT(
            hidden_size,
            factor=3,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> torch.Tensor:
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
            img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk
            q = torch.cat((img_q, txt_q), dim=1)
            k = torch.cat((img_k, txt_k), dim=1)

        # Compute attention.
        assert (
            cu_seqlens_q.shape[0] == 2 * x.shape[0] + 1
        ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, x.shape[0]:{x.shape[0]}"
        
        # attention computation start
        if not self.hybrid_seq_parallel_attn:
            attn = attention(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=x.shape[0],
            )
        else:
            attn = parallel_attention(
                self.hybrid_seq_parallel_attn,
                q,
                k,
                v,
                img_q_len=img_q.shape[1],
                img_kv_len=img_k.shape[1],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv
            )
        # attention computation end

        # Compute activation in mlp stream, cat again and run second linear layer.
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + apply_gate(output, gate=mod_gate)


class ResidualDoubleAdapter(nn.Module):
    def __init__(self, hidden_size: int, rank: int, scale: float):
        super().__init__()
        self.scale = scale

        self.img_norm = nn.LayerNorm(hidden_size)
        self.txt_norm = nn.LayerNorm(hidden_size)

        self.img_down = nn.Linear(hidden_size, rank, bias=False)
        self.img_up = nn.Linear(rank, hidden_size, bias=False)
        self.txt_down = nn.Linear(hidden_size, rank, bias=False)
        self.txt_up = nn.Linear(rank, hidden_size, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.img_down.weight)
        nn.init.zeros_(self.img_up.weight)
        nn.init.zeros_(self.txt_down.weight)
        nn.init.zeros_(self.txt_up.weight)

    def forward(
        self, img: torch.Tensor, txt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_delta = self.img_up(self.img_down(self.img_norm(img))) * self.scale
        txt_delta = self.txt_up(self.txt_down(self.txt_norm(txt))) * self.scale
        energy = (img_delta.pow(2).mean() + txt_delta.pow(2).mean())
        return img + img_delta, txt + txt_delta, energy


class ResidualSingleAdapter(nn.Module):
    def __init__(self, hidden_size: int, rank: int, scale: float):
        super().__init__()
        self.scale = scale

        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        delta = self.up(self.down(self.norm(x))) * self.scale
        energy = delta.pow(2).mean()
        return x + delta, energy


class HYVideoDiffusionTransformer(ModelMixin, ConfigMixin):
    """
    HunyuanVideo Transformer backbone

    Inherited from ModelMixin and ConfigMixin for compatibility with diffusers' sampler StableDiffusionPipeline.

    Reference:
    [1] Flux.1: https://github.com/black-forest-labs/flux
    [2] MMDiT: http://arxiv.org/abs/2403.03206

    Parameters
    ----------
    args: argparse.Namespace
        The arguments parsed by argparse.
    patch_size: list
        The size of the patch.
    in_channels: int
        The number of input channels.
    out_channels: int
        The number of output channels.
    hidden_size: int
        The hidden size of the transformer backbone.
    heads_num: int
        The number of attention heads.
    mlp_width_ratio: float
        The ratio of the hidden size of the MLP in the transformer block.
    mlp_act_type: str
        The activation function of the MLP in the transformer block.
    depth_double_blocks: int
        The number of transformer blocks in the double blocks.
    depth_single_blocks: int
        The number of transformer blocks in the single blocks.
    rope_dim_list: list
        The dimension of the rotary embedding for t, h, w.
    qkv_bias: bool
        Whether to use bias in the qkv linear layer.
    qk_norm: bool
        Whether to use qk norm.
    qk_norm_type: str
        The type of qk norm.
    guidance_embed: bool
        Whether to use guidance embedding for distillation.
    text_projection: str
        The type of the text projection, default is single_refiner.
    use_attention_mask: bool
        Whether to use attention mask for text encoder.
    dtype: torch.dtype
        The dtype of the model.
    device: torch.device
        The device of the model.
    """

    @register_to_config
    def __init__(
        self,
        args: Any,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,  # Should be VAE.config.latent_channels.
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: List[int] = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,  # For modulation.
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list

        # Layer offloading configuration
        self.layer_offload = bool(getattr(args, "layer_offload", False))
        self.prefetch_offload = (
            bool(getattr(args, "prefetch_offload", False)) and torch.cuda.is_available()
        )
        self.int8_cache_offload = (
            bool(getattr(args, "int8_cache_offload", False)) and torch.cuda.is_available()
        )
        quant_mode = getattr(args, "int8_quant_mode", None)
        if quant_mode not in {"channel", "tensor"}:
            quant_mode = "tensor" if getattr(args, "int8_per_tensor", False) else "channel"
        self.int8_quant_mode = quant_mode
        self.int8_per_channel = self.int8_quant_mode != "tensor"

        raw_offload_blocks = getattr(args, "offload_blocks", "")
        self.offload_blocks = self._parse_offload_blocks(raw_offload_blocks)

        # Weight sharing configuration (experimental memory compression)
        total_block_count = mm_double_blocks_depth + mm_single_blocks_depth
        raw_share_map = getattr(args, "layer_share_map", "")
        self.layer_share_map: Dict[int, int] = {}
        self._double_share_map: Dict[int, int] = {}
        self._single_share_map: Dict[int, int] = {}
        if isinstance(raw_share_map, str) and raw_share_map.strip():
            share_pairs = [pair.strip() for pair in raw_share_map.split(",") if pair.strip()]
            for pair in share_pairs:
                if "->" in pair:
                    target_str, source_str = pair.split("->", 1)
                elif ":" in pair:
                    target_str, source_str = pair.split(":", 1)
                else:
                    raise ValueError(
                        "Invalid format for --layer-share-map. Use 'target->source' pairs separated by commas."
                    )

                try:
                    target_idx = int(target_str.strip())
                    source_idx = int(source_str.strip())
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid block index in --layer-share-map entry '{pair}'."
                    ) from exc

                if not (0 <= target_idx < total_block_count) or not (0 <= source_idx < total_block_count):
                    raise ValueError(
                        f"Block indices out of range in --layer-share-map entry '{pair}'."
                    )
                if target_idx == source_idx:
                    raise ValueError(
                        f"Cannot map block {target_idx} to itself in --layer-share-map."
                    )
                if target_idx in self.layer_share_map:
                    raise ValueError(
                        f"Duplicate target index {target_idx} in --layer-share-map."
                    )
                if source_idx > target_idx:
                    raise ValueError(
                        f"Source index must precede target index for '{pair}' to avoid recursive ties."
                    )

                target_is_double = target_idx < mm_double_blocks_depth
                source_is_double = source_idx < mm_double_blocks_depth
                if target_is_double != source_is_double:
                    raise ValueError(
                        f"Cannot share weights across double/single block types for entry '{pair}'."
                    )

                if target_is_double:
                    local_target = target_idx
                    local_source = source_idx
                    if local_source >= mm_double_blocks_depth:
                        raise ValueError(
                            f"Source index {source_idx} invalid for double block entry '{pair}'."
                        )
                    if local_source >= local_target:
                        raise ValueError(
                            f"Double block targets must map to an earlier block for entry '{pair}'."
                        )
                    self._double_share_map[local_target] = local_source
                else:
                    local_target = target_idx - mm_double_blocks_depth
                    local_source = source_idx - mm_double_blocks_depth
                    if local_target >= mm_single_blocks_depth or local_source < 0:
                        raise ValueError(
                            f"Source index {source_idx} invalid for single block entry '{pair}'."
                        )
                    if local_source >= local_target:
                        raise ValueError(
                            f"Single block targets must map to an earlier block for entry '{pair}'."
                        )
                    self._single_share_map[local_target] = local_source

                self.layer_share_map[target_idx] = source_idx
        elif raw_share_map:
            raise ValueError("--layer-share-map must be provided as a string of mappings.")

        # Text projection. Default to linear projection.
        # Alternative: TokenRefiner. See more details (LI-DiT): http://arxiv.org/abs/2406.11831
        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection

        self.text_states_dim = args.text_states_dim
        self.text_states_dim_2 = args.text_states_dim_2

        if hidden_size % heads_num != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}"
            )
        pe_dim = hidden_size // heads_num
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(
                f"Got {rope_dim_list} but expected positional dim {pe_dim}"
            )
        self.hidden_size = hidden_size
        self.heads_num = heads_num

        # image projection
        self.img_in = PatchEmbed(
            self.patch_size, self.in_channels, self.hidden_size, **factory_kwargs
        )

        # text projection
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                self.text_states_dim,
                self.hidden_size,
                get_activation_layer("silu"),
                **factory_kwargs,
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(
                self.text_states_dim, hidden_size, heads_num, depth=2, **factory_kwargs
            )
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        # time modulation
        self.time_in = TimestepEmbedder(
            self.hidden_size, get_activation_layer("silu"), **factory_kwargs
        )

        # text modulation
        self.vector_in = MLPEmbedder(
            self.text_states_dim_2, self.hidden_size, **factory_kwargs
        )

        # guidance modulation
        self.guidance_in = (
            TimestepEmbedder(
                self.hidden_size, get_activation_layer("silu"), **factory_kwargs
            )
            if guidance_embed
            else None
        )

        # double blocks
        double_blocks: List[MMDoubleStreamBlock] = []
        for idx in range(mm_double_blocks_depth):
            share_source = self._double_share_map.get(idx)
            if share_source is not None:
                double_blocks.append(double_blocks[share_source])
            else:
                double_blocks.append(
                    MMDoubleStreamBlock(
                        self.hidden_size,
                        self.heads_num,
                        mlp_width_ratio=mlp_width_ratio,
                        mlp_act_type=mlp_act_type,
                        qk_norm=qk_norm,
                        qk_norm_type=qk_norm_type,
                        qkv_bias=qkv_bias,
                        **factory_kwargs,
                    )
                )
        self.double_blocks = nn.ModuleList(double_blocks)

        # single blocks
        single_blocks: List[MMSingleStreamBlock] = []
        for idx in range(mm_single_blocks_depth):
            share_source = self._single_share_map.get(idx)
            if share_source is not None:
                single_blocks.append(single_blocks[share_source])
            else:
                single_blocks.append(
                    MMSingleStreamBlock(
                        self.hidden_size,
                        self.heads_num,
                        mlp_width_ratio=mlp_width_ratio,
                        mlp_act_type=mlp_act_type,
                        qk_norm=qk_norm,
                        qk_norm_type=qk_norm_type,
                        **factory_kwargs,
                    )
                )
        self.single_blocks = nn.ModuleList(single_blocks)

        default_device = self._normalize_device(device)
        self._int8_cache: Dict[int, Dict[str, Any]] = {}

        for block in self.double_blocks:
            block._hy_device = default_device
        for block in self.single_blocks:
            block._hy_device = default_device

        self.share_adapter_rank = int(getattr(args, "share_adapter_rank", 0) or 0)
        self.share_adapter_scale = float(getattr(args, "share_adapter_scale", 1.0))
        self.share_adapters = nn.ModuleDict()
        if self.share_adapter_rank > 0 and self.layer_share_map:
            for target_idx in sorted(self.layer_share_map.keys()):
                if target_idx < len(self.double_blocks):
                    adapter = ResidualDoubleAdapter(
                        self.hidden_size,
                        self.share_adapter_rank,
                        self.share_adapter_scale,
                    )
                else:
                    adapter = ResidualSingleAdapter(
                        self.hidden_size,
                        self.share_adapter_rank,
                        self.share_adapter_scale,
                    )
                self.share_adapters[str(target_idx)] = adapter
                adapter._hy_device = default_device
        self._share_adapter_indices = {int(k) for k in self.share_adapters.keys()}
        self._adapter_last_energy: Optional[float] = None

        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            get_activation_layer("silu"),
            **factory_kwargs,
        )

    @staticmethod
    def _parse_offload_blocks(raw_offload_blocks: Any) -> set:
        """Parse offload block indices from a string or iterable."""
        offload_blocks: set = set()
        if isinstance(raw_offload_blocks, str) and raw_offload_blocks.strip():
            for token in raw_offload_blocks.split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    offload_blocks.add(int(token))
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid block index '{token}' received in --offload-blocks"
                    ) from exc
        elif isinstance(raw_offload_blocks, (list, tuple, set)):
            for token in raw_offload_blocks:
                try:
                    offload_blocks.add(int(token))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid block index '{token}' received in --offload-blocks"
                    ) from exc
        return offload_blocks

    def enable_deterministic(self):
        for block in self.double_blocks:
            block.enable_deterministic()
        for block in self.single_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        for block in self.double_blocks:
            block.disable_deterministic()
        for block in self.single_blocks:
            block.disable_deterministic()

    def _normalize_device(self, device: Optional[Union[str, torch.device]]) -> torch.device:
        if isinstance(device, torch.device):
            return device
        if device is None:
            return torch.device("cpu")
        return torch.device(device)

    def _move_block_to_device(
        self,
        block: nn.Module,
        device: Union[str, torch.device],
        non_blocking: bool = False,
        pin_memory: bool = False,
    ) -> None:
        target_device = self._normalize_device(device)
        current_device = getattr(block, "_hy_device", None)
        if current_device == target_device:
            return

        non_blocking = bool(non_blocking and target_device.type == "cuda")
        pin_memory = bool(pin_memory and target_device.type == "cpu")

        cache = self._int8_cache.get(id(block)) if self.int8_cache_offload else None
        use_cache = (
            cache is not None
            and target_device.type == "cuda"
        )

        if use_cache:
            cache_device = cache.get("_device")
            if cache_device is None or cache_device != target_device:
                for key, entry in list(cache.items()):
                    if key == "_device":
                        continue
                    entry["tensor"] = entry["tensor"].to(target_device, non_blocking=True)
                    entry["scale"] = entry["scale"].to(target_device, non_blocking=True)
                cache["_device"] = target_device

        with torch.no_grad():
            if use_cache:
                for name, param in block.named_parameters():
                    entry = cache.get(name)
                    if entry is None:
                        data = param.data.to(target_device, non_blocking=non_blocking)
                    else:
                        q_tensor = entry["tensor"].to(dtype=torch.float32)
                        scale = entry["scale"]
                        data = (q_tensor * scale).to(param.dtype)
                    param.data = data
            else:
                for param in block.parameters():
                    data = param.data.to(target_device, non_blocking=non_blocking)
                    if pin_memory:
                        data = data.pin_memory()
                    param.data = data

            for buffer in block.buffers():
                if buffer is None or not torch.is_tensor(buffer):
                    continue
                data = buffer.data.to(target_device, non_blocking=non_blocking)
                if pin_memory:
                    data = data.pin_memory()
                buffer.data = data

        block._hy_device = target_device

    def _move_block_to_cpu(
        self,
        block: nn.Module,
        cache_device: Optional[torch.device] = None,
    ) -> None:
        if (
            self.int8_cache_offload
            and cache_device is not None
            and cache_device.type == "cuda"
        ):
            self._ensure_int8_cache(block, cache_device)

        self._move_block_to_device(
            block,
            torch.device("cpu"),
            non_blocking=False,
            pin_memory=self.prefetch_offload,
        )

    def _ensure_int8_cache(
        self, block: nn.Module, device: torch.device
    ) -> None:
        cache = self._int8_cache.setdefault(id(block), {})
        cache_device = cache.get("_device")
        if cache_device is not None and cache_device != device:
            cache.clear()

        with torch.no_grad():
            for name, param in block.named_parameters():
                if name in cache:
                    continue
                param_fp32 = param.data.detach().to(torch.float32)
                max_abs = param_fp32.abs().amax()
                scale = torch.clamp(max_abs / 127.0, min=1e-8)
                quantized = torch.round(param_fp32 / scale).clamp(-127, 127).to(torch.int8)
                cache[name] = {
                    "tensor": quantized.to(device, non_blocking=True),
                    "scale": scale.to(device, non_blocking=True),
                }

        cache["_device"] = device

    def _find_next_offload_index(self, start_idx: int) -> Optional[int]:
        total_blocks = len(self.double_blocks) + len(self.single_blocks)
        for idx in range(start_idx, total_blocks):
            if idx in self.offload_blocks:
                return idx
        return None

    def _get_block_by_global_index(self, global_idx: int) -> nn.Module:
        if global_idx < len(self.double_blocks):
            return self.double_blocks[global_idx]
        single_idx = global_idx - len(self.double_blocks)
        if single_idx < 0 or single_idx >= len(self.single_blocks):
            raise IndexError(f"Invalid block index: {global_idx}")
        return self.single_blocks[single_idx]

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # Should be in range(0, 1000).
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,  # Now we don't use it.
        text_states_2: Optional[torch.Tensor] = None,  # Text embedding for modulation.
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        guidance: torch.Tensor = None,  # Guidance for modulation, should be cfg_scale x 1000.
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        out = {}
        img = x
        txt = text_states
        _, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )

        # Prepare modulation vectors.
        vec = self.time_in(t)

        # text modulation
        vec = vec + self.vector_in(text_states_2)

        # guidance modulation
        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )

            # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
            vec = vec + self.guidance_in(guidance)

        # Embed image and text.
        img = self.img_in(img)
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        # Compute cu_squlens and max_seqlen for flash attention
        cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        # --------------------- Pass through DiT blocks ------------------------
        prefetch_enabled = (
            self.layer_offload
            and self.prefetch_offload
            and img.device.type == "cuda"
        )
        device_index = (
            img.device.index
            if img.device.type == "cuda" and img.device.index is not None
            else (torch.cuda.current_device() if prefetch_enabled else None)
        )
        prefetch_stream = None
        if prefetch_enabled:
            try:
                prefetch_stream = torch.cuda.Stream(device=device_index)
            except RuntimeError as exc:
                logger.warning("Failed to create CUDA stream for prefetching: %s", exc)
                prefetch_enabled = False
        if prefetch_enabled and prefetch_stream is None:
            prefetch_enabled = False

        pending_prefetch: Optional[Tuple[int, nn.Module]] = None
        adapter_energy: Optional[torch.Tensor] = None

        if prefetch_enabled and self.offload_blocks:
            first_offload_idx = self._find_next_offload_index(0)
            if first_offload_idx is not None:
                first_block = self._get_block_by_global_index(first_offload_idx)
                if getattr(first_block, "_hy_device", None) != img.device:
                    try:
                        if prefetch_stream is not None:
                            with torch.cuda.stream(prefetch_stream):
                                self._move_block_to_device(
                                    first_block, img.device, non_blocking=True
                                )
                        else:
                            self._move_block_to_device(first_block, img.device)
                        pending_prefetch = (first_offload_idx, first_block)
                    except RuntimeError as exc:
                        logger.warning(
                            "Initial prefetch failed for block %s: %s",
                            first_offload_idx,
                            exc,
                        )
                        pending_prefetch = None
                        prefetch_enabled = False
                        prefetch_stream = None

        for idx, block in enumerate(self.double_blocks):
            global_idx = idx
            should_offload = self.layer_offload and global_idx in self.offload_blocks

            # Layer offloading: move block to GPU before forward pass
            if should_offload:
                if (
                    prefetch_enabled
                    and prefetch_stream is not None
                    and pending_prefetch
                    and pending_prefetch[0] == global_idx
                ):
                    torch.cuda.current_stream(device=device_index).wait_stream(
                        prefetch_stream
                    )
                    pending_prefetch = None
                elif getattr(block, "_hy_device", None) != img.device:
                    self._move_block_to_device(block, img.device)

            double_block_args = [
                img,
                txt,
                vec,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                freqs_cis,
            ]

            img, txt = block(*double_block_args)

            if self._share_adapter_indices and global_idx in self._share_adapter_indices:
                adapter = self.share_adapters[str(global_idx)]
                adapter_device = getattr(adapter, "_hy_device", None)
                if adapter_device != img.device:
                    adapter.to(img.device)
                    adapter._hy_device = img.device
                img, txt, energy = adapter(img, txt)
                adapter_energy = energy if adapter_energy is None else adapter_energy + energy

            # Layer offloading: move block back to CPU after forward pass
            if should_offload:
                if prefetch_enabled and prefetch_stream is not None:
                    next_idx = self._find_next_offload_index(global_idx + 1)
                    if next_idx is not None:
                        next_block = self._get_block_by_global_index(next_idx)
                        if getattr(next_block, "_hy_device", None) != img.device:
                            with torch.cuda.stream(prefetch_stream):
                                self._move_block_to_device(
                                    next_block, img.device, non_blocking=True
                                )
                            pending_prefetch = (next_idx, next_block)
                        else:
                            pending_prefetch = None
                    else:
                        pending_prefetch = None
                elif prefetch_enabled:
                    next_idx = self._find_next_offload_index(global_idx + 1)
                    if next_idx is not None:
                        next_block = self._get_block_by_global_index(next_idx)
                        if getattr(next_block, "_hy_device", None) != img.device:
                            self._move_block_to_device(next_block, img.device)
                        pending_prefetch = None
                    else:
                        pending_prefetch = None

                cache_device = img.device if img.device.type == "cuda" else None
                self._move_block_to_cpu(block, cache_device)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Merge txt and img to pass through single stream blocks.
        x = torch.cat((img, txt), 1)
        if len(self.single_blocks) > 0:
            for idx, block in enumerate(self.single_blocks):
                # Layer offloading for single blocks (offset by double blocks count)
                single_idx = len(self.double_blocks) + idx
                should_offload = self.layer_offload and single_idx in self.offload_blocks

                if should_offload:
                    if (
                        prefetch_enabled
                        and prefetch_stream is not None
                        and pending_prefetch
                        and pending_prefetch[0] == single_idx
                    ):
                        torch.cuda.current_stream(device=device_index).wait_stream(
                            prefetch_stream
                        )
                        pending_prefetch = None
                    elif getattr(block, "_hy_device", None) != x.device:
                        self._move_block_to_device(block, x.device)

                single_block_args = [
                    x,
                    vec,
                    txt_seq_len,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    (freqs_cos, freqs_sin),
                ]

                x = block(*single_block_args)

                if self._share_adapter_indices and single_idx in self._share_adapter_indices:
                    adapter = self.share_adapters[str(single_idx)]
                    adapter_device = getattr(adapter, "_hy_device", None)
                    if adapter_device != x.device:
                        adapter.to(x.device)
                        adapter._hy_device = x.device
                    x, energy = adapter(x)
                    adapter_energy = energy if adapter_energy is None else adapter_energy + energy

                # Layer offloading: move block back to CPU after forward pass
                if should_offload:
                    if prefetch_enabled and prefetch_stream is not None:
                        next_idx = self._find_next_offload_index(single_idx + 1)
                        if next_idx is not None:
                            next_block = self._get_block_by_global_index(next_idx)
                            if getattr(next_block, "_hy_device", None) != x.device:
                                with torch.cuda.stream(prefetch_stream):
                                    self._move_block_to_device(
                                        next_block, x.device, non_blocking=True
                                    )
                                pending_prefetch = (next_idx, next_block)
                            else:
                                pending_prefetch = None
                        else:
                            pending_prefetch = None
                    elif prefetch_enabled:
                        next_idx = self._find_next_offload_index(single_idx + 1)
                        if next_idx is not None:
                            next_block = self._get_block_by_global_index(next_idx)
                            if getattr(next_block, "_hy_device", None) != x.device:
                                self._move_block_to_device(next_block, x.device)
                            pending_prefetch = None
                        else:
                            pending_prefetch = None

                    cache_device = x.device if x.device.type == "cuda" else None
                    self._move_block_to_cpu(block, cache_device)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        img = x[:, :img_seq_len, ...]

        if adapter_energy is not None:
            self._adapter_last_energy = adapter_energy.detach().cpu().item()
        else:
            self._adapter_last_energy = 0.0

        # ---------------------------- Final layer ------------------------------
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

        img = self.unpatchify(img, tt, th, tw)
        if return_dict:
            out["x"] = img
            return out
        return img

    def unpatchify(self, x, t, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))

        return imgs

    def params_count(self):
        def _unique_param_sum(blocks, attr_names):
            seen = set()
            total = 0
            for block in blocks:
                block_id = id(block)
                if block_id in seen:
                    continue
                seen.add(block_id)
                for attr in attr_names:
                    layer = getattr(block, attr, None)
                    if layer is None:
                        continue
                    total += sum(p.numel() for p in layer.parameters())
            return total

        double_param_count = _unique_param_sum(
            self.double_blocks,
            [
                "img_attn_qkv",
                "img_attn_proj",
                "img_mlp",
                "txt_attn_qkv",
                "txt_attn_proj",
                "txt_mlp",
            ],
        )
        single_param_count = _unique_param_sum(
            self.single_blocks,
            ["linear1", "linear2"],
        )

        counts = {
            "double": double_param_count,
            "single": single_param_count,
            "total": sum(p.numel() for p in self.parameters()),
        }
        counts["attn+mlp"] = counts["double"] + counts["single"]
        return counts


#################################################################################
#                             HunyuanVideo Configs                              #
#################################################################################

HUNYUAN_VIDEO_CONFIG = {
    "HYVideo-T/2": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
    },
    "HYVideo-T/2-cfgdistill": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
        "guidance_embed": True,
    },
}
