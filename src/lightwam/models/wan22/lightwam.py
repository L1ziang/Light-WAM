import time
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from lightwam.utils.logging_config import get_logger
from lightwam.utils.pytorch_utils import synchronize_device

from .action_dit import ActionDiT
from .helpers.loader import (
    apply_video_backbone_preset,
    load_wan_video_components,
    resolve_video_backbone_type,
    sync_action_dit_config_with_video_backbone,
)
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .state_fusion_action_expert import StateFusionActionExpert

logger = get_logger(__name__)


class DisabledActionExpert(nn.Module):
    """Zero-parameter placeholder used when the old action expert is fully removed."""

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = int(action_dim)

    def pre_dit(self, *args, **kwargs):
        raise RuntimeError("The original action expert is disabled in state-fusion adapter mode.")

    def post_dit(self, *args, **kwargs):
        raise RuntimeError("The original action expert is disabled in state-fusion adapter mode.")


class LightWAM(torch.nn.Module):
    """Canonical Light-WAM policy model with Wan video backbone and state-fusion action decoding."""

    def __init__(
        self,
        video_expert,
        action_expert: nn.Module,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_backbone_type: str = "wan2_2_ti2v",
        video_latent_spatial_downsample_factor: int = 1,
        apply_video_latent_downsample_to_action_branch: bool = False,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        use_first_frame_residual_video_target: bool = False,
        action_temporal_weighting_enabled: bool = False,
        action_temporal_weighting_num_prefix_steps: Optional[int] = None,
        action_temporal_weighting_prefix_weight: float = 1.0,
        action_temporal_weighting_tail_weight: float = 1.0,
        use_wam_adapter: bool = False,
        freeze_backbone: bool = True,
        remove_original_action_expert: bool = False,
        state_fusion_action_expert: Optional[StateFusionActionExpert] = None,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.video_backbone_type = str(video_backbone_type)
        self.video_latent_spatial_downsample_factor = int(video_latent_spatial_downsample_factor)
        if self.video_latent_spatial_downsample_factor < 1:
            raise ValueError(
                "`video_latent_spatial_downsample_factor` must be >= 1, "
                f"got {self.video_latent_spatial_downsample_factor}."
            )
        self.apply_video_latent_downsample_to_action_branch = bool(
            apply_video_latent_downsample_to_action_branch
        )
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.use_first_frame_residual_video_target = bool(use_first_frame_residual_video_target)
        self.action_temporal_weighting_enabled = bool(action_temporal_weighting_enabled)
        self.action_temporal_weighting_num_prefix_steps = (
            None
            if action_temporal_weighting_num_prefix_steps is None
            else int(action_temporal_weighting_num_prefix_steps)
        )
        self.action_temporal_weighting_prefix_weight = float(
            action_temporal_weighting_prefix_weight
        )
        self.action_temporal_weighting_tail_weight = float(action_temporal_weighting_tail_weight)
        if self.action_temporal_weighting_enabled:
            if (
                self.action_temporal_weighting_num_prefix_steps is None
                or self.action_temporal_weighting_num_prefix_steps <= 0
            ):
                raise ValueError(
                    "`action_temporal_weighting_num_prefix_steps` must be positive when "
                    "`action_temporal_weighting_enabled=true`."
                )
            if self.action_temporal_weighting_prefix_weight <= 0.0:
                raise ValueError(
                    "`action_temporal_weighting_prefix_weight` must be > 0, "
                    f"got {self.action_temporal_weighting_prefix_weight}."
                )
            if self.action_temporal_weighting_tail_weight < 0.0:
                raise ValueError(
                    "`action_temporal_weighting_tail_weight` must be >= 0, "
                    f"got {self.action_temporal_weighting_tail_weight}."
                )
        self.use_wam_adapter = bool(use_wam_adapter)
        self.freeze_backbone = bool(freeze_backbone)
        self.remove_original_action_expert = bool(remove_original_action_expert)
        self.state_fusion_action_expert = state_fusion_action_expert
        self.enable_timing_breakdown = False
        self.timing_breakdown_sync_cuda = True
        self._timing_breakdown: dict[str, float] = {}
        if self.remove_original_action_expert and not self.use_wam_adapter:
            raise ValueError(
                "`remove_original_action_expert=true` requires `use_wam_adapter=true`."
            )

        self.to(self.device)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        video_backbone_type: str = "wan2_2_ti2v",
        video_backbone_name: str | None = None,
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        video_latent_spatial_downsample_factor: int = 1,
        apply_video_latent_downsample_to_action_branch: bool = False,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        use_first_frame_residual_video_target: bool = False,
        action_temporal_weighting_enabled: bool = False,
        action_temporal_weighting_num_prefix_steps: Optional[int] = None,
        action_temporal_weighting_prefix_weight: float = 1.0,
        action_temporal_weighting_tail_weight: float = 1.0,
        wam_adapter: dict[str, Any] | None = None,
        state_fusion_action_expert_config: dict[str, Any] | None = None,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for LightWAM.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for Light-WAM.")
        resolved_video_backbone_type = resolve_video_backbone_type(video_backbone_type)
        video_dit_config = apply_video_backbone_preset(
            dit_config=dict(video_dit_config),
            video_backbone_type=resolved_video_backbone_type,
        )
        action_dit_config = {} if action_dit_config is None else dict(action_dit_config)
        action_dit_config = sync_action_dit_config_with_video_backbone(
            action_dit_config=action_dit_config,
            video_dit_config=video_dit_config,
        )
        wam_adapter_cfg = {} if wam_adapter is None else dict(wam_adapter)
        state_fusion_action_expert_config = (
            {} if state_fusion_action_expert_config is None else dict(state_fusion_action_expert_config)
        )
        use_wam_adapter = bool(wam_adapter_cfg.get("use_wam_adapter", False))
        freeze_backbone = bool(wam_adapter_cfg.get("freeze_backbone", True))
        remove_original_action_expert = bool(
            wam_adapter_cfg.get("remove_original_action_expert", False)
        )
        use_backbone_lora = bool(wam_adapter_cfg.get("use_backbone_lora", False))
        if remove_original_action_expert and not use_wam_adapter:
            raise ValueError(
                "`wam_adapter.remove_original_action_expert=true` requires `wam_adapter.use_wam_adapter=true`."
            )
        use_state_fusion_action_expert = use_wam_adapter and remove_original_action_expert
        if use_wam_adapter:
            video_dit_config["use_wam_adapter"] = True
            video_dit_config["adapter_layer_indices"] = wam_adapter_cfg.get("adapter_layer_indices")
            video_dit_config["adapter_dim"] = int(wam_adapter_cfg.get("adapter_dim", 128))
            video_dit_config["adapter_scale"] = float(wam_adapter_cfg.get("adapter_scale", 1.0))
            if not use_state_fusion_action_expert:
                action_dit_config["use_wam_adapter"] = True
                action_dit_config["video_hidden_dim"] = int(video_dit_config["hidden_dim"])
            else:
                action_dit_config.pop("use_wam_adapter", None)
                action_dit_config.pop("video_hidden_dim", None)
        if use_backbone_lora:
            video_dit_config["use_backbone_lora"] = True
            video_dit_config["lora_layer_indices"] = wam_adapter_cfg.get("lora_layer_indices")
            video_dit_config["lora_target_modules"] = wam_adapter_cfg.get(
                "lora_target_modules",
                ["ffn.0", "ffn.2"],
            )
            video_dit_config["lora_rank"] = int(wam_adapter_cfg.get("lora_rank", 16))
            video_dit_config["lora_alpha"] = float(wam_adapter_cfg.get("lora_alpha", 16.0))
            video_dit_config["lora_dropout"] = float(wam_adapter_cfg.get("lora_dropout", 0.0))

        components = load_wan_video_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            video_backbone_type=resolved_video_backbone_type,
            video_backbone_name=video_backbone_name,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        if use_state_fusion_action_expert:
            # In state-fusion mode the original ActionDiT is fully removed, so it does not
            # allocate GPU memory and the action path runs only through the new direct expert.
            action_expert = DisabledActionExpert(action_dim=int(action_dit_config["action_dim"]))
            mot = MoT(
                mixtures={"video": video_expert},
                mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            )
        else:
            action_expert = ActionDiT.from_pretrained(
                action_dit_config=action_dit_config,
                action_dit_pretrained_path=action_dit_pretrained_path,
                skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
                device=device,
                torch_dtype=torch_dtype,
            )
            if int(action_expert.num_heads) != int(video_expert.num_heads):
                raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
            if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
                raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
            if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
                raise ValueError("ActionDiT `num_layers` must match video expert.")
            mot = MoT(
                mixtures={"video": video_expert, "action": action_expert},
                mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            )
        state_fusion_action_expert = None
        if use_state_fusion_action_expert:
            num_fusion_layers = int(len(getattr(video_expert, "adapter_layer_indices", ())))
            if num_fusion_layers <= 0:
                raise ValueError(
                    "State-fusion adapter mode requires at least one configured adapter layer."
                )
            state_fusion_action_expert = StateFusionActionExpert(
                video_hidden_dim=int(video_dit_config["hidden_dim"]),
                action_dim=int(action_dit_config["action_dim"]),
                num_fusion_layers=num_fusion_layers,
                **state_fusion_action_expert_config,
            ).to(device=device, dtype=torch_dtype)

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_backbone_type=resolved_video_backbone_type,
            video_latent_spatial_downsample_factor=video_latent_spatial_downsample_factor,
            apply_video_latent_downsample_to_action_branch=(
                apply_video_latent_downsample_to_action_branch
            ),
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            use_first_frame_residual_video_target=use_first_frame_residual_video_target,
            action_temporal_weighting_enabled=action_temporal_weighting_enabled,
            action_temporal_weighting_num_prefix_steps=action_temporal_weighting_num_prefix_steps,
            action_temporal_weighting_prefix_weight=action_temporal_weighting_prefix_weight,
            action_temporal_weighting_tail_weight=action_temporal_weighting_tail_weight,
            use_wam_adapter=use_wam_adapter,
            freeze_backbone=freeze_backbone,
            remove_original_action_expert=remove_original_action_expert,
            state_fusion_action_expert=state_fusion_action_expert,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "NOT_INSTANTIATED_IN_STATE_FUSION_ACTION_EXPERT_MODE"
                if use_state_fusion_action_expert
                else ("SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path)
            ),
        }
        return model

    def configure_trainable_modules(self):
        """Apply adapter-aware train/freeze policy while preserving the legacy default path."""
        self.eval()
        self.requires_grad_(False)
        has_backbone_lora = (
            hasattr(self.video_expert, "has_backbone_lora")
            and self.video_expert.has_backbone_lora()
        )

        if self.uses_state_fusion_action_expert():
            logger.info(
                "Setting state-fusion adapter mode: freezing the backbone and old action expert, "
                "training adapters, future head, and the new state_fusion_action_expert."
            )
            self.video_expert.train()
            if has_backbone_lora:
                self.video_expert.enable_backbone_lora_training()
            if hasattr(self.video_expert, "wam_adapters"):
                self.video_expert.wam_adapters.train()
                self.video_expert.wam_adapters.requires_grad_(True)
            # Future/video supervision still adapts through the original video head.
            self.video_expert.head.train()
            self.video_expert.head.requires_grad_(True)
            if self.state_fusion_action_expert is None:
                raise RuntimeError("`state_fusion_action_expert` is required in state-fusion mode.")
            self.state_fusion_action_expert.train()
            self.state_fusion_action_expert.requires_grad_(True)
            self.action_expert.eval()
        elif (self.use_wam_adapter or has_backbone_lora) and self.freeze_backbone:
            logger.info(
                "Setting adapter/LoRA PEFT mode: freezing video backbone and training backbone PEFT modules, action expert, and future/action heads."
            )
            self.mot.train()
            self.action_expert.train()
            self.action_expert.requires_grad_(True)
            if has_backbone_lora:
                self.video_expert.enable_backbone_lora_training()
            if hasattr(self.video_expert, "wam_adapters"):
                self.video_expert.wam_adapters.train()
                self.video_expert.wam_adapters.requires_grad_(True)
            # The video head remains trainable so future-token supervision can adapt on top of frozen blocks.
            self.video_expert.head.train()
            self.video_expert.head.requires_grad_(True)
        else:
            logger.info("Setting DiT to train mode and freezing other model components.")
            self.mot.train()
            self.mot.requires_grad_(True)

        proprio_encoder = getattr(self, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)

    def uses_state_fusion_action_expert(self) -> bool:
        return bool(
            self.use_wam_adapter
            and self.remove_original_action_expert
            and self.state_fusion_action_expert is not None
        )

    def set_timing_breakdown(self, enabled: bool, sync_cuda: bool = True):
        self.enable_timing_breakdown = bool(enabled)
        self.timing_breakdown_sync_cuda = bool(sync_cuda)
        self._timing_breakdown = {}

    def _timing_sync(self):
        if not self.enable_timing_breakdown:
            return
        if self.timing_breakdown_sync_cuda:
            synchronize_device(self.device)

    def _timing_start(self) -> Optional[float]:
        if not self.enable_timing_breakdown:
            return None
        self._timing_sync()
        return time.perf_counter()

    def _timing_end(self, name: str, start_time: Optional[float]):
        if start_time is None:
            return
        self._timing_sync()
        duration_s = time.perf_counter() - start_time
        self._timing_breakdown[name] = self._timing_breakdown.get(name, 0.0) + duration_s

    def _reset_timing_breakdown(self):
        self._timing_breakdown = {}

    def _get_timing_breakdown_metrics(self) -> dict[str, float]:
        return {
            f"timing/model/{name}_ms": float(duration_s * 1000.0)
            for name, duration_s in sorted(self._timing_breakdown.items())
        }

    def _benchmark_sync_device(self) -> None:
        synchronize_device(self.device)

    def _prepare_infer_action_context(
        self,
        prompt: Optional[str],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            prepared_context, prepared_context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            prepared_context = context.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
            prepared_context_mask = context_mask.to(
                device=self.device,
                dtype=torch.bool,
                non_blocking=True,
            )

        if proprio is not None:
            prepared_context, prepared_context_mask = self._append_proprio_to_context(
                context=prepared_context,
                context_mask=prepared_context_mask,
                proprio=proprio,
            )
        return prepared_context, prepared_context_mask

    def _maybe_downsample_video_latents_for_backbone(
        self,
        latents: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[dict[str, Any]]]:
        """Optionally compress full-video latent grids before the video backbone.

        This is intended for Wan2.1-T2V style backbones where the latent grid is much
        denser than Wan2.2-TI2V. Default factor=1 keeps the original Fast-WAM path
        exactly unchanged.
        """
        factor = int(self.video_latent_spatial_downsample_factor)
        if factor == 1:
            return latents, None
        if latents.ndim != 5:
            raise ValueError(
                f"`latents` must be [B, C, T, H, W], got shape {tuple(latents.shape)}"
            )
        height = int(latents.shape[-2])
        width = int(latents.shape[-1])
        if height % factor != 0 or width % factor != 0:
            raise ValueError(
                "Latent spatial shape must be divisible by "
                f"`video_latent_spatial_downsample_factor={factor}`, "
                f"got HxW=({height}, {width})."
            )
        latents_down = F.avg_pool3d(
            latents,
            kernel_size=(1, factor, factor),
            stride=(1, factor, factor),
        )
        return latents_down, {
            "original_spatial_shape": (height, width),
            "downsample_factor": factor,
        }

    def _restore_video_prediction_spatial_resolution(
        self,
        pred_video: torch.Tensor,
        compression_meta: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        if compression_meta is None:
            return pred_video
        if pred_video.ndim != 5:
            raise ValueError(
                f"`pred_video` must be [B, C, T, H, W], got shape {tuple(pred_video.shape)}"
            )
        original_spatial_shape = compression_meta.get("original_spatial_shape")
        if original_spatial_shape is None or len(original_spatial_shape) != 2:
            raise ValueError("`compression_meta['original_spatial_shape']` must be a 2-tuple.")
        original_height = int(original_spatial_shape[0])
        original_width = int(original_spatial_shape[1])
        if pred_video.shape[-2:] == (original_height, original_width):
            return pred_video

        batch_size, channels, num_frames, _, _ = pred_video.shape
        pred_video_btc = pred_video.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames,
            channels,
            pred_video.shape[-2],
            pred_video.shape[-1],
        )
        pred_video_btc = F.interpolate(
            pred_video_btc,
            size=(original_height, original_width),
            mode="bilinear",
            align_corners=False,
        )
        return pred_video_btc.reshape(
            batch_size,
            num_frames,
            channels,
            original_height,
            original_width,
        ).permute(0, 2, 1, 3, 4).contiguous()

    def _build_video_pre(
        self,
        latents_video: torch.Tensor,
        timestep_video: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        action: Optional[torch.Tensor] = None,
        apply_spatial_downsample: bool = True,
    ) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
        compression_meta = None
        latents_for_backbone = latents_video
        if apply_spatial_downsample:
            latents_for_backbone, compression_meta = self._maybe_downsample_video_latents_for_backbone(
                latents_video
            )
        video_pre = self.video_expert.pre_dit(
            x=latents_for_backbone,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        return video_pre, compression_meta

    def _build_action_observation_video_pre(
        self,
        observation_latents: torch.Tensor,
        timestep_video: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> dict[str, Any]:
        # The single-frame action observation path stays high resolution by default.
        video_pre, _ = self._build_video_pre(
            latents_video=observation_latents,
            timestep_video=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            apply_spatial_downsample=self.apply_video_latent_downsample_to_action_branch,
        )
        return video_pre

    def _use_lowres_video_training_objective(self) -> bool:
        return int(self.video_latent_spatial_downsample_factor) > 1

    def _prepare_video_training_targets(
        self,
        video_supervision_latents: torch.Tensor,
        timestep_video: torch.Tensor,
        first_frame_latents: Optional[torch.Tensor],
    ) -> dict[str, Any]:
        video_supervision_latents_model = video_supervision_latents
        first_frame_latents_model = first_frame_latents
        apply_spatial_downsample = True
        restore_spatial_resolution = True

        if self._use_lowres_video_training_objective():
            # Keep the full-video future objective in the same low-resolution latent space
            # that the backbone actually sees when `video_latent_spatial_downsample_factor > 1`.
            video_supervision_latents_model, _ = self._maybe_downsample_video_latents_for_backbone(
                video_supervision_latents
            )
            if first_frame_latents is not None:
                first_frame_latents_model, _ = self._maybe_downsample_video_latents_for_backbone(
                    first_frame_latents
                )
            apply_spatial_downsample = False
            restore_spatial_resolution = False

        noise_video = torch.randn_like(video_supervision_latents_model)
        latents_video = self.train_video_scheduler.add_noise(
            video_supervision_latents_model,
            noise_video,
            timestep_video,
        )
        target_video = self.train_video_scheduler.training_target(
            video_supervision_latents_model,
            noise_video,
            timestep_video,
        )
        if first_frame_latents_model is not None:
            latents_video[:, :, 0:1] = first_frame_latents_model

        return {
            "video_supervision_latents_model": video_supervision_latents_model,
            "first_frame_latents_model": first_frame_latents_model,
            "latents_video": latents_video,
            "target_video": target_video,
            "apply_spatial_downsample": apply_spatial_downsample,
            "restore_spatial_resolution": restore_spatial_resolution,
        }

    def _estimate_clean_video_latents(
        self,
        noisy_latents: torch.Tensor,
        flow_prediction: torch.Tensor,
        timestep_video: torch.Tensor,
    ) -> torch.Tensor:
        sigma = (timestep_video / float(self.train_video_scheduler.num_train_timesteps)).to(
            noisy_latents.device,
            dtype=noisy_latents.dtype,
        )
        sigma = sigma.view(-1, *([1] * (noisy_latents.ndim - 1)))
        return noisy_latents - sigma * flow_prediction

    def _decode_video_tokens(
        self,
        video_tokens: torch.Tensor,
        video_pre: dict[str, Any],
        compression_meta: Optional[dict[str, Any]],
        restore_spatial_resolution: bool = True,
    ) -> torch.Tensor:
        pred_video = self.video_expert.post_dit(video_tokens, video_pre)
        if not restore_spatial_resolution:
            return pred_video
        return self._restore_video_prediction_spatial_resolution(pred_video, compression_meta)

    @staticmethod
    def _count_module_parameters(module: Optional[nn.Module]) -> tuple[int, int]:
        if module is None:
            return 0, 0
        total = 0
        trainable = 0
        for param in module.parameters():
            total += int(param.numel())
            if param.requires_grad:
                trainable += int(param.numel())
        return total, trainable

    @staticmethod
    def _count_module_list_parameters(modules: Sequence[Optional[nn.Module]]) -> tuple[int, int]:
        seen: set[int] = set()
        total = 0
        trainable = 0
        for module in modules:
            if module is None:
                continue
            for param in module.parameters():
                param_id = id(param)
                if param_id in seen:
                    continue
                seen.add(param_id)
                total += int(param.numel())
                if param.requires_grad:
                    trainable += int(param.numel())
        return total, trainable

    @staticmethod
    def _format_param_count(value: int) -> str:
        if value >= 1_000_000_000:
            return f"{value:,} ({value / 1_000_000_000:.3f}B)"
        if value >= 1_000_000:
            return f"{value:,} ({value / 1_000_000:.3f}M)"
        if value >= 1_000:
            return f"{value:,} ({value / 1_000:.3f}K)"
        return str(value)

    def log_parameter_summary(self):
        logger.info("Model parameter summary:")
        logger.info("  use_wam_adapter=%s", self.use_wam_adapter)
        logger.info("  freeze_backbone=%s", self.freeze_backbone)
        logger.info("  remove_original_action_expert=%s", self.remove_original_action_expert)
        if hasattr(self.video_expert, "has_backbone_lora") and self.video_expert.has_backbone_lora():
            logger.info(
                "  backbone_lora: layers=%s targets=%s rank=%s alpha=%s dropout=%s",
                list(getattr(self.video_expert, "lora_layer_indices", ())),
                list(getattr(self.video_expert, "lora_target_modules", ())),
                getattr(self.video_expert, "lora_rank", None),
                getattr(self.video_expert, "lora_alpha", None),
                getattr(self.video_expert, "lora_dropout", None),
            )
        logger.info("  video_backbone_type=%s", self.video_backbone_type)
        logger.info(
            "  video_latent_spatial_downsample_factor=%s",
            self.video_latent_spatial_downsample_factor,
        )
        logger.info(
            "  apply_video_latent_downsample_to_action_branch=%s",
            self.apply_video_latent_downsample_to_action_branch,
        )
        logger.info(
            "  action_temporal_weighting: enabled=%s prefix_steps=%s prefix_weight=%s tail_weight=%s",
            self.action_temporal_weighting_enabled,
            self.action_temporal_weighting_num_prefix_steps,
            self.action_temporal_weighting_prefix_weight,
            self.action_temporal_weighting_tail_weight,
        )
        if self.state_fusion_action_expert is not None:
            logger.info(
                "  state_fusion.token_pooling_type=%s token_pooling_num_queries=%s token_pooling_num_heads=%s token_pooling_merge_dim=%s token_pooling_merge_num_slots=%s",
                self.state_fusion_action_expert.token_pooling_type,
                self.state_fusion_action_expert.token_pooling_num_queries,
                self.state_fusion_action_expert.token_pooling_num_heads,
                self.state_fusion_action_expert.token_pooling_merge_dim,
                self.state_fusion_action_expert.token_pooling_merge_num_slots,
            )
            logger.info(
                "  state_fusion.layer_feature_sources=%s",
                [list(source_names) for source_names in self.state_fusion_action_expert.layer_feature_sources],
            )

        major_modules = {
            "video_expert": self.video_expert,
            "mot": self.mot,
            "action_expert": self.action_expert,
            "state_fusion_action_expert": self.state_fusion_action_expert,
            "proprio_encoder": self.proprio_encoder,
        }
        for name, module in major_modules.items():
            total, trainable = self._count_module_parameters(module)
            logger.info(
                "  module=%s total=%s trainable=%s",
                name,
                self._format_param_count(total),
                self._format_param_count(trainable),
            )

        video_backbone_total, video_backbone_trainable = self._count_module_list_parameters(
            [
                getattr(self.video_expert, "patch_embedding", None),
                getattr(self.video_expert, "text_embedding", None),
                getattr(self.video_expert, "time_embedding", None),
                getattr(self.video_expert, "time_projection", None),
                getattr(self.video_expert, "blocks", None),
            ]
        )
        adapters_total, adapters_trainable = self._count_module_parameters(
            getattr(self.video_expert, "wam_adapters", None)
        )
        backbone_lora_total, backbone_lora_trainable = self._count_module_list_parameters(
            (
                getattr(self.video_expert, "get_backbone_lora_modules", lambda: [])()
                if hasattr(self.video_expert, "get_backbone_lora_modules")
                else []
            )
        )
        future_head_total, future_head_trainable = self._count_module_parameters(
            getattr(self.video_expert, "head", None)
        )
        old_action_total, old_action_trainable = self._count_module_parameters(self.action_expert)
        new_action_total, new_action_trainable = self._count_module_parameters(
            self.state_fusion_action_expert
        )
        proprio_total, proprio_trainable = self._count_module_parameters(self.proprio_encoder)

        group_rows = [
            ("video/WAM backbone", video_backbone_total, video_backbone_trainable, "frozen" if self.freeze_backbone else "trainable"),
            ("adapters", adapters_total, adapters_trainable, "enabled" if self.use_wam_adapter else "disabled"),
            (
                "backbone LoRA",
                backbone_lora_total,
                backbone_lora_trainable,
                (
                    "enabled"
                    if hasattr(self.video_expert, "has_backbone_lora") and self.video_expert.has_backbone_lora()
                    else "disabled"
                ),
            ),
            ("future head", future_head_total, future_head_trainable, "trainable"),
            (
                "old action expert",
                old_action_total,
                old_action_trainable,
                (
                    "not_instantiated"
                    if self.uses_state_fusion_action_expert() and old_action_total == 0
                    else ("bypassed+frozen" if self.uses_state_fusion_action_expert() else "active")
                ),
            ),
            (
                "new state_fusion_action_expert",
                new_action_total,
                new_action_trainable,
                "active" if self.uses_state_fusion_action_expert() else "disabled",
            ),
            ("proprio encoder", proprio_total, proprio_trainable, "active" if self.proprio_encoder is not None else "disabled"),
        ]
        for name, total, trainable, status in group_rows:
            logger.info(
                "  group=%s total=%s trainable=%s status=%s",
                name,
                self._format_param_count(total),
                self._format_param_count(trainable),
                status,
            )

        trainable_module_names = sorted(
            {
                name.rsplit(".", 1)[0] if "." in name else name
                for name, param in self.named_parameters()
                if param.requires_grad
            }
        )
        logger.info("  trainable_modules=%s", trainable_module_names)
        total_params, trainable_params = self._count_module_parameters(self)
        logger.info("  total_params=%s", self._format_param_count(total_params))
        logger.info("  total_trainable_params=%s", self._format_param_count(trainable_params))

    @staticmethod
    def _pool_video_tokens(tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(
                f"Expected video tokens for fusion to be [B, S, D], got {tuple(tokens.shape)}"
            )
        return tokens.mean(dim=1)

    def _build_action_fusion(
        self,
        video_tokens: Optional[torch.Tensor] = None,
        video_token_slice: Optional[slice] = None,
    ) -> Optional[torch.Tensor]:
        if not self.use_wam_adapter:
            return None
        backbone_tokens, adapted_tokens = self.video_expert.get_wam_action_fusion_states(
            fallback_tokens=video_tokens,
        )
        if video_token_slice is not None:
            backbone_tokens = backbone_tokens[:, video_token_slice]
            adapted_tokens = adapted_tokens[:, video_token_slice]
        if tuple(backbone_tokens.shape) != tuple(adapted_tokens.shape):
            raise ValueError(
                "Backbone/action fusion tokens must match, "
                f"got {tuple(backbone_tokens.shape)} and {tuple(adapted_tokens.shape)}"
            )
        delta_tokens = adapted_tokens - backbone_tokens
        return torch.cat(
            [
                self._pool_video_tokens(backbone_tokens),
                self._pool_video_tokens(adapted_tokens),
                self._pool_video_tokens(delta_tokens),
            ],
            dim=-1,
        )

    def _build_multilayer_action_fusion_inputs(
        self,
        video_token_slice: Optional[slice] = None,
    ) -> list[dict[str, Any]]:
        if not self.uses_state_fusion_action_expert():
            raise RuntimeError(
                "`_build_multilayer_action_fusion_inputs` requires state-fusion action mode."
            )
        layer_states = self.video_expert.get_wam_action_fusion_layer_states(
            selected_layers=list(getattr(self.video_expert, "adapter_layer_indices", ())),
        )
        fusion_inputs: list[dict[str, Any]] = []
        for layer_idx, backbone_tokens, adapted_tokens in layer_states:
            if video_token_slice is not None:
                backbone_tokens = backbone_tokens[:, video_token_slice]
                adapted_tokens = adapted_tokens[:, video_token_slice]
            if tuple(backbone_tokens.shape) != tuple(adapted_tokens.shape):
                raise ValueError(
                    f"Layer {layer_idx} fusion state mismatch: "
                    f"{tuple(backbone_tokens.shape)} vs {tuple(adapted_tokens.shape)}"
                )
            delta_tokens = adapted_tokens - backbone_tokens
            fusion_inputs.append(
                {
                    "layer_idx": int(layer_idx),
                    "backbone": backbone_tokens,
                    "adapted": adapted_tokens,
                    "delta": delta_tokens,
                }
            )
        return fusion_inputs

    def _decode_action_tokens(
        self,
        action_tokens: torch.Tensor,
        action_pre: dict[str, Any],
        video_tokens: Optional[torch.Tensor] = None,
        video_token_slice: Optional[slice] = None,
    ) -> torch.Tensor:
        if not self.use_wam_adapter:
            return self.action_expert.post_dit(action_tokens, action_pre)
        action_pre = dict(action_pre)
        # The action path sees both the frozen video state and the adapter residual correction.
        action_pre["action_fusion"] = self._build_action_fusion(
            video_tokens=video_tokens,
            video_token_slice=video_token_slice,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        timing_start = self._timing_start()
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        self._timing_end("vae_encode", timing_start)
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(self, sample, tiled: bool = False):
        timing_start = self._timing_start()
        video = sample.get("video", None)
        video_latents = sample.get("video_latents", None)
        if video is None and video_latents is None:
            raise ValueError(
                "Light-WAM training requires either `sample['video']` or `sample['video_latents']`."
            )
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "Light-WAM training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        image_is_pad = sample.get("image_is_pad", None)

        if video_latents is not None:
            # Optional fast path for latent-cache training: reuse precomputed VAE latents
            # and keep the downstream Light-WAM future/action logic unchanged.
            if not isinstance(video_latents, torch.Tensor):
                raise TypeError(
                    f"`sample['video_latents']` must be a torch.Tensor, got {type(video_latents)}"
                )
            if video_latents.ndim == 4:
                video_latents = video_latents.unsqueeze(0)
            if video_latents.ndim != 5:
                raise ValueError(
                    "`sample['video_latents']` must be 5D [B, C, T, H, W], "
                    f"got shape {tuple(video_latents.shape)}"
                )
            batch_size, latent_channels, latent_t, _, _ = video_latents.shape
            expected_latent_channels = int(getattr(self.vae, "z_dim", self.vae.model.z_dim))
            if latent_channels != expected_latent_channels:
                raise ValueError(
                    "`sample['video_latents']` channel mismatch: "
                    f"got {latent_channels}, expected {expected_latent_channels}"
                )
            input_latents = video_latents.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            if image_is_pad is not None:
                if image_is_pad.ndim != 2:
                    raise ValueError(
                        f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                    )
                if image_is_pad.shape[0] != batch_size:
                    raise ValueError(
                        "`sample['image_is_pad']` batch mismatch with `video_latents`: "
                        f"{tuple(image_is_pad.shape)} vs batch_size={batch_size}"
                    )
                num_frames = int(image_is_pad.shape[1])
                expected_latent_t = ((num_frames - 1) // int(self.vae.temporal_downsample_factor)) + 1
                if latent_t != expected_latent_t:
                    raise ValueError(
                        "`sample['video_latents']` temporal mismatch with `image_is_pad`: "
                        f"latent_t={latent_t}, expected={expected_latent_t}, num_frames={num_frames}"
                    )
            else:
                num_frames = int((latent_t - 1) * int(self.vae.temporal_downsample_factor) + 1)
        else:
            if not isinstance(video, torch.Tensor):
                raise TypeError(f"`sample['video']` must be a torch.Tensor, got {type(video)}")
            if video.ndim != 5:
                raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
            if video.shape[1] != 3:
                raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

            batch_size, _, num_frames, height, width = video.shape
            if height % 16 != 0 or width % 16 != 0:
                raise ValueError(
                    f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
                )
            if num_frames % 4 != 1:
                raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
            if num_frames <= 1:
                raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for Light-WAM training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )

        if video_latents is None:
            input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            input_latents = self._encode_video_latents(input_video, tiled=tiled)

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio = proprio[:, 0, :] # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        self._timing_end("build_inputs", timing_start)
        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        # action -> first-frame video only
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def _build_video_training_supervision_latents(self, input_latents: torch.Tensor) -> torch.Tensor:
        if not self.use_first_frame_residual_video_target:
            return input_latents
        first_frame_latents = input_latents[:, :, 0:1]
        return input_latents - first_frame_latents

    def _build_loss_dict(self, loss_video: torch.Tensor, loss_action: torch.Tensor) -> dict[str, float]:
        loss_video_raw = float(loss_video.detach().item())
        loss_action_raw = float(loss_action.detach().item())
        return {
            "loss_video": self.loss_lambda_video * loss_video_raw,
            "loss_action": self.loss_lambda_action * loss_action_raw,
            "loss_video_raw": loss_video_raw,
            "loss_action_raw": loss_action_raw,
        }

    def _compute_action_loss_per_sample(
        self,
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        action_loss_token = F.mse_loss(
            pred_action.float(),
            target_action.float(),
            reduction="none",
        ).mean(dim=2)
        step_weights = self._build_action_temporal_weights(
            action_horizon=int(action_loss_token.shape[1]),
            device=action_loss_token.device,
            dtype=action_loss_token.dtype,
        ).view(1, -1)
        if action_is_pad is None:
            weight_sum = step_weights.sum(dim=1).clamp(min=1e-6)
            return (action_loss_token * step_weights).sum(dim=1) / weight_sum
        valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
        weighted_valid = valid * step_weights
        valid_sum = weighted_valid.sum(dim=1).clamp(min=1e-6)
        return (action_loss_token * weighted_valid).sum(dim=1) / valid_sum

    def _build_action_temporal_weights(
        self,
        action_horizon: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")
        weights = torch.ones((action_horizon,), device=device, dtype=dtype)
        if not self.action_temporal_weighting_enabled:
            return weights
        prefix_steps = min(int(self.action_temporal_weighting_num_prefix_steps), action_horizon)
        weights.fill_(self.action_temporal_weighting_tail_weight)
        weights[:prefix_steps] = self.action_temporal_weighting_prefix_weight
        return weights

    def _predict_video_only(
        self,
        latents_video: torch.Tensor,
        timestep_video: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        action: Optional[torch.Tensor] = None,
        apply_spatial_downsample: bool = True,
        restore_spatial_resolution: bool = True,
    ) -> torch.Tensor:
        total_timing = self._timing_start()
        video_pre, compression_meta = self._build_video_pre(
            latents_video=latents_video,
            timestep_video=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            apply_spatial_downsample=apply_spatial_downsample,
        )
        backbone_timing = self._timing_start()
        video_tokens = self.video_expert.forward_backbone(video_pre)
        self._timing_end("future_backbone", backbone_timing)
        decode_timing = self._timing_start()
        pred_video = self._decode_video_tokens(
            video_tokens,
            video_pre,
            compression_meta,
            restore_spatial_resolution=restore_spatial_resolution,
        )
        self._timing_end("future_decode", decode_timing)
        self._timing_end("future_total", total_timing)
        return pred_video

    def _predict_state_fusion_action_from_observation(
        self,
        observation_latents: torch.Tensor,
        action_horizon: int,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        if not self.uses_state_fusion_action_expert():
            raise RuntimeError(
                "`_predict_state_fusion_action_from_observation` requires state-fusion action mode."
            )
        if self.state_fusion_action_expert is None:
            raise RuntimeError("`state_fusion_action_expert` is not initialized.")
        if observation_latents.ndim != 5 or observation_latents.shape[2] != 1:
            raise ValueError(
                "`observation_latents` must be a single-frame latent tensor [B, C, 1, H, W], "
                f"got {tuple(observation_latents.shape)}"
            )

        total_timing = self._timing_start()
        timestep_video = torch.zeros(
            (observation_latents.shape[0],),
            dtype=observation_latents.dtype,
            device=observation_latents.device,
        )
        video_pre = self._build_action_observation_video_pre(
            observation_latents=observation_latents,
            timestep_video=timestep_video,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        # The new action expert consumes multi-layer pooled [h, h', delta] features from the
        # single-observation backbone pass, matching the direct-action inference path.
        backbone_timing = self._timing_start()
        _ = self.video_expert.forward_backbone(video_pre)
        self._timing_end("action_backbone", backbone_timing)
        expert_timing = self._timing_start()
        pred_action = self.state_fusion_action_expert(
            self._build_multilayer_action_fusion_inputs(),
            action_horizon=action_horizon,
        )
        self._timing_end("state_fusion_action_expert", expert_timing)
        if pred_action.ndim != 3:
            raise ValueError(
                f"`state_fusion_action_expert` must return [B, T, A], got {tuple(pred_action.shape)}"
            )
        if pred_action.shape[0] != observation_latents.shape[0] or pred_action.shape[1] != action_horizon:
            raise ValueError(
                "State-fusion action output shape mismatch: "
                f"got {tuple(pred_action.shape)} for batch={observation_latents.shape[0]} "
                f"and action_horizon={action_horizon}"
            )
        self._timing_end("action_total", total_timing)
        return pred_action

    def _training_loss_state_fusion(self, sample, tiled: bool = False):
        self._reset_timing_breakdown()
        total_timing = self._timing_start()
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        video_supervision_latents = self._build_video_training_supervision_latents(input_latents)
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]
        fuse_flag = inputs["fuse_vae_embedding_in_latents"]

        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=video_supervision_latents.dtype,
        )
        video_train_targets = self._prepare_video_training_targets(
            video_supervision_latents=video_supervision_latents,
            timestep_video=timestep_video,
            first_frame_latents=inputs["first_frame_latents"],
        )
        latents_video = video_train_targets["latents_video"]
        target_video = video_train_targets["target_video"]

        pred_video = self._predict_video_only(
            latents_video=latents_video,
            timestep_video=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=fuse_flag,
            apply_spatial_downsample=video_train_targets["apply_spatial_downsample"],
            restore_spatial_resolution=video_train_targets["restore_spatial_resolution"],
        )

        observation_latents = inputs["first_frame_latents"]
        if observation_latents is None:
            observation_latents = input_latents[:, :, 0:1]
        pred_action = self._predict_state_fusion_action_from_observation(
            observation_latents=observation_latents,
            action_horizon=int(action.shape[1]),
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_flag,
        )

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device,
            dtype=loss_video_per_sample.dtype,
        )
        loss_video = (loss_video_per_sample * video_weight).mean()
        loss_action = self._compute_action_loss_per_sample(
            pred_action=pred_action,
            target_action=action,
            action_is_pad=action_is_pad,
        ).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = self._build_loss_dict(loss_video=loss_video, loss_action=loss_action)
        self._timing_end("training_loss_total", total_timing)
        if self.enable_timing_breakdown:
            loss_dict.update(self._get_timing_breakdown_metrics())
        return loss_total, loss_dict

    def training_loss(self, sample, tiled: bool = False):
        if self.uses_state_fusion_action_expert():
            return self._training_loss_state_fusion(sample, tiled=tiled)

        self._reset_timing_breakdown()
        total_timing = self._timing_start()
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        video_supervision_latents = self._build_video_training_supervision_latents(input_latents)
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=video_supervision_latents.dtype,
        )
        video_train_targets = self._prepare_video_training_targets(
            video_supervision_latents=video_supervision_latents,
            timestep_video=timestep_video,
            first_frame_latents=inputs["first_frame_latents"],
        )
        latents = video_train_targets["latents_video"]
        target_video = video_train_targets["target_video"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre, compression_meta = self._build_video_pre(
            latents_video=latents,
            timestep_video=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            apply_spatial_downsample=video_train_targets["apply_spatial_downsample"],
        )

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_tokens,
                "action": action_tokens,
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self._decode_video_tokens(
            tokens_out["video"],
            video_pre,
            compression_meta,
            restore_spatial_resolution=video_train_targets["restore_spatial_resolution"],
        )

        pred_action = self._decode_action_tokens(
            action_tokens=tokens_out["action"],
            action_pre=action_pre,
            video_tokens=tokens_out["video"],
        )

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_per_sample = self._compute_action_loss_per_sample(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=action_is_pad,
        )
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = self._build_loss_dict(loss_video=loss_video, loss_action=loss_action)
        self._timing_end("training_loss_total", total_timing)
        if self.enable_timing_breakdown:
            loss_dict.update(self._get_timing_breakdown_metrics())
        return loss_total, loss_dict

    @torch.no_grad()
    def render_training_visualization(
        self,
        sample,
        tiled: bool = False,
    ) -> dict[str, Any]:
        """Render a lightweight prediction-vs-target training visualization.

        This path runs under `no_grad` on a tiny batch and immediately decodes to CPU
        frames so it does not retain extra training activations in GPU memory.
        """
        was_training = self.training
        self.eval()
        try:
            inputs = self.build_inputs(sample, tiled=tiled)
            input_latents = inputs["input_latents"]
            video_supervision_latents = self._build_video_training_supervision_latents(input_latents)
            batch_size = int(input_latents.shape[0])
            if batch_size < 1:
                raise ValueError("Visualization requires at least one sample in the batch.")

            timestep_video = self.train_video_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=video_supervision_latents.dtype,
            )
            video_train_targets = self._prepare_video_training_targets(
                video_supervision_latents=video_supervision_latents,
                timestep_video=timestep_video,
                first_frame_latents=inputs["first_frame_latents"],
            )

            pred_video = self._predict_video_only(
                latents_video=video_train_targets["latents_video"],
                timestep_video=timestep_video,
                context=inputs["context"],
                context_mask=inputs["context_mask"],
                action=inputs["action"],
                fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
                apply_spatial_downsample=video_train_targets["apply_spatial_downsample"],
                restore_spatial_resolution=video_train_targets["restore_spatial_resolution"],
            )
            pred_clean = self._estimate_clean_video_latents(
                noisy_latents=video_train_targets["latents_video"],
                flow_prediction=pred_video,
                timestep_video=timestep_video,
            )
            target_clean = video_train_targets["video_supervision_latents_model"].clone()
            if video_train_targets["first_frame_latents_model"] is not None:
                pred_clean[:, :, 0:1] = video_train_targets["first_frame_latents_model"]
                target_clean[:, :, 0:1] = video_train_targets["first_frame_latents_model"]

            pred_frames = self._decode_latents(pred_clean[:1], tiled=tiled)
            target_frames = self._decode_latents(target_clean[:1], tiled=tiled)

            stitched_frames = []
            for pred_frame, target_frame in zip(pred_frames, target_frames):
                pred_arr = np.array(pred_frame.convert("RGB"))
                target_arr = np.array(target_frame.convert("RGB"))
                if pred_arr.shape[0] != target_arr.shape[0]:
                    raise ValueError(
                        "Visualization frame height mismatch: "
                        f"pred={pred_arr.shape} vs target={target_arr.shape}"
                    )
                stitched_frames.append(
                    Image.fromarray(np.concatenate([pred_arr, target_arr], axis=1))
                )

            return {
                "frames": stitched_frames,
                "caption": "pred | target",
            }
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_pre, compression_meta = self._build_video_pre(
            latents_video=latents_video,
            timestep_video=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )

        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        # Adapter mode keeps the original Fast-WAM objective and swaps in the adapted video states.
        pred_video = self._decode_video_tokens(tokens_out["video"], video_pre, compression_meta)
        pred_action = self._decode_action_tokens(
            action_tokens=tokens_out["action"],
            action_pre=action_pre,
            video_tokens=tokens_out["video"],
        )
        return pred_video, pred_action

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self._build_action_observation_video_pre(
            observation_latents=first_frame_latents,
            timestep_video=timestep_video,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_action = self._decode_action_tokens(
            action_tokens=tokens_out["action"],
            action_pre=action_pre,
            video_tokens=tokens_out["video"],
        )
        return pred_action

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self._decode_action_tokens(
            action_tokens=action_tokens,
            action_pre=action_pre,
        )

    @torch.no_grad()
    def _infer_joint_state_fusion(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
    ) -> dict[str, Any]:
        del negative_prompt, text_cfg_scale
        self.eval()

        action_only_out = None
        if test_action_with_infer_action:
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
            )["action"]

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        pred_action = self._predict_state_fusion_action_from_observation(
            observation_latents=first_frame_latents,
            action_horizon=action_horizon,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_flag,
        )

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video in zip(infer_timesteps_video, infer_deltas_video):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            pred_video = self._predict_video_only(
                latents_video=latents_video,
                timestep_video=timestep_video,
                context=context,
                context_mask=context_mask,
                action=action,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = pred_action[0].detach().to(device="cpu", dtype=torch.float32)
        if action_only_out is not None and not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
            max_abs_diff = (action_out - action_only_out).abs().max().item()
            logger.warning(
                "Action from infer_joint and infer_action differ with max abs diff %.6f.",
                max_abs_diff,
            )
        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None, # NOTE: this is gt action for conditioning videos, not for action expert
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
    ) -> dict[str, Any]:
        if self.uses_state_fusion_action_expert():
            return self._infer_joint_state_fusion(
                prompt=prompt,
                input_image=input_image,
                num_video_frames=num_video_frames,
                action_horizon=action_horizon,
                action=action,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                negative_prompt=negative_prompt,
                text_cfg_scale=text_cfg_scale,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                test_action_with_infer_action=test_action_with_infer_action,
            )

        self.eval()
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
            )["action"]
        
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        if self.uses_state_fusion_action_expert():
            del negative_prompt, text_cfg_scale, num_inference_steps, sigma_shift, seed, rand_device
            input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
            first_frame_latents = self._encode_input_image_latents_tensor(
                input_image=input_image,
                tiled=tiled,
            )
            fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

            use_prompt = prompt is not None
            use_context = context is not None or context_mask is not None
            if use_prompt and use_context:
                raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
            if not use_prompt and not use_context:
                raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

            if use_prompt:
                context, context_mask = self.encode_prompt(prompt)
            else:
                if context is None or context_mask is None:
                    raise ValueError("`context` and `context_mask` must be both provided together.")
                if context.ndim == 2:
                    context = context.unsqueeze(0)
                if context_mask.ndim == 1:
                    context_mask = context_mask.unsqueeze(0)
                if context.ndim != 3 or context_mask.ndim != 2:
                    raise ValueError(
                        f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                    )
                context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
                context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
            if proprio is not None:
                context, context_mask = self._append_proprio_to_context(
                    context=context,
                    context_mask=context_mask,
                    proprio=proprio,
                )

            pred_action = self._predict_state_fusion_action_from_observation(
                observation_latents=first_frame_latents,
                action_horizon=action_horizon,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            return {
                "action": pred_action[0].detach().to(device="cpu", dtype=torch.float32),
            }

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self._build_action_observation_video_pre(
            observation_latents=first_frame_latents,
            timestep_video=timestep_video,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action_posi = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def benchmark_infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        del negative_prompt, text_cfg_scale

        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`benchmark_infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        timings_s: dict[str, float] = {}
        self._benchmark_sync_device()
        total_start = time.perf_counter()

        vae_start = time.perf_counter()
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(
            input_image=input_image,
            tiled=tiled,
        )
        self._benchmark_sync_device()
        timings_s["vae_encode"] = float(time.perf_counter() - vae_start)

        condition_start = time.perf_counter()
        prepared_context, prepared_context_mask = self._prepare_infer_action_context(
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
        )
        self._benchmark_sync_device()
        timings_s["condition_prepare"] = float(time.perf_counter() - condition_start)

        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        if self.uses_state_fusion_action_expert():
            model_start = time.perf_counter()
            timestep_video = torch.zeros(
                (first_frame_latents.shape[0],),
                dtype=first_frame_latents.dtype,
                device=self.device,
            )

            prepare_start = time.perf_counter()
            video_pre = self._build_action_observation_video_pre(
                observation_latents=first_frame_latents,
                timestep_video=timestep_video,
                context=prepared_context,
                context_mask=prepared_context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            self._benchmark_sync_device()
            timings_s["model_prepare_observation"] = float(time.perf_counter() - prepare_start)

            backbone_start = time.perf_counter()
            _ = self.video_expert.forward_backbone(video_pre)
            self._benchmark_sync_device()
            timings_s["model_action_backbone"] = float(time.perf_counter() - backbone_start)

            head_start = time.perf_counter()
            pred_action = self.state_fusion_action_expert(
                self._build_multilayer_action_fusion_inputs(),
                action_horizon=action_horizon,
            )
            self._benchmark_sync_device()
            timings_s["model_state_fusion_action_head"] = float(time.perf_counter() - head_start)
            timings_s["model_predict"] = float(time.perf_counter() - model_start)
            self._benchmark_sync_device()
            timings_s["total"] = float(time.perf_counter() - total_start)
            return {
                "action": pred_action[0].detach().to(device="cpu", dtype=torch.float32),
                "timings_s": timings_s,
                "uses_state_fusion_action_expert": True,
            }

        model_start = time.perf_counter()
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        prefill_start = time.perf_counter()
        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self._build_action_observation_video_pre(
            observation_latents=first_frame_latents,
            timestep_video=timestep_video,
            context=prepared_context,
            context_mask=prepared_context_mask,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )
        self._benchmark_sync_device()
        timings_s["model_video_prefill_cache"] = float(time.perf_counter() - prefill_start)

        denoise_start = time.perf_counter()
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action_posi = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=prepared_context,
                context_mask=prepared_context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            latents_action = self.infer_action_scheduler.step(
                pred_action_posi,
                step_delta_action,
                latents_action,
            )
        self._benchmark_sync_device()
        timings_s["model_action_denoise_loop"] = float(time.perf_counter() - denoise_start)
        timings_s["model_predict"] = float(time.perf_counter() - model_start)
        self._benchmark_sync_device()
        timings_s["total"] = float(time.perf_counter() - total_start)
        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
            "timings_s": timings_s,
            "uses_state_fusion_action_expert": False,
        }

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ):
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.state_fusion_action_expert is not None:
            payload["state_fusion_action_expert"] = self.state_fusion_action_expert.state_dict()
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        logger.info(
            "Loading Light-WAM checkpoint: path=%s step=%s keys=%s",
            path,
            payload.get("step", None),
            sorted(payload.keys()),
        )
        if "mot" in payload:
            if not isinstance(payload["mot"], dict):
                raise TypeError(f"Checkpoint `mot` entry must be a state_dict-like dict, got {type(payload['mot'])}")
            model_state = self.mot.state_dict()
            checkpoint_state = payload["mot"]
            overlap_keys = set(model_state.keys()) & set(checkpoint_state.keys())
            if not overlap_keys:
                raise ValueError(
                    f"Checkpoint `mot` has zero overlapping keys with current model: {path}"
                )
            incompatible = self.mot.load_state_dict(checkpoint_state, strict=False)
            self._log_load_state_dict_result(
                module_name="mot",
                model_state_dict=model_state,
                checkpoint_state_dict=checkpoint_state,
                incompatible_keys=incompatible,
            )
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            if not isinstance(payload["dit"], dict):
                raise TypeError(f"Checkpoint `dit` entry must be a state_dict-like dict, got {type(payload['dit'])}")
            model_state = self.video_expert.state_dict()
            checkpoint_state = payload["dit"]
            overlap_keys = set(model_state.keys()) & set(checkpoint_state.keys())
            if not overlap_keys:
                raise ValueError(
                    f"Legacy checkpoint `dit` has zero overlapping keys with current video expert: {path}"
                )
            incompatible = self.video_expert.load_state_dict(checkpoint_state, strict=False)
            self._log_load_state_dict_result(
                module_name="video_expert(legacy_dit)",
                model_state_dict=model_state,
                checkpoint_state_dict=checkpoint_state,
                incompatible_keys=incompatible,
            )
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
                logger.info("Checkpoint load summary [proprio_encoder]: strict load succeeded.")
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")
        if self.state_fusion_action_expert is not None:
            if "state_fusion_action_expert" in payload:
                model_state = self.state_fusion_action_expert.state_dict()
                checkpoint_state = payload["state_fusion_action_expert"]
                overlap_keys = set(model_state.keys()) & set(checkpoint_state.keys())
                if not overlap_keys:
                    raise ValueError(
                        f"Checkpoint `state_fusion_action_expert` has zero overlapping keys with current model: {path}"
                    )
                filtered_checkpoint_state, skipped_shape_keys = self._filter_compatible_state_dict(
                    model_state_dict=model_state,
                    checkpoint_state_dict=checkpoint_state,
                )
                if not filtered_checkpoint_state:
                    raise ValueError(
                        f"Checkpoint `state_fusion_action_expert` has no shape-compatible overlapping keys with current model: {path}"
                    )
                incompatible = self.state_fusion_action_expert.load_state_dict(
                    filtered_checkpoint_state,
                    strict=False,
                )
                self._log_load_state_dict_result(
                    module_name="state_fusion_action_expert",
                    model_state_dict=model_state,
                    checkpoint_state_dict=filtered_checkpoint_state,
                    incompatible_keys=incompatible,
                )
                if skipped_shape_keys:
                    logger.warning(
                        "Checkpoint load skipped shape-mismatched keys [state_fusion_action_expert] (showing up to 10): %s",
                        skipped_shape_keys[:10],
                    )
            else:
                logger.warning(
                    "Checkpoint has no `state_fusion_action_expert` weights; keeping current direct action expert params."
                )
        elif "state_fusion_action_expert" in payload:
            logger.warning(
                "Checkpoint contains `state_fusion_action_expert` weights but current model does not enable it; ignoring."
            )

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    @staticmethod
    def _log_load_state_dict_result(
        module_name: str,
        model_state_dict: dict[str, torch.Tensor],
        checkpoint_state_dict: dict[str, torch.Tensor],
        incompatible_keys,
    ):
        model_keys = set(model_state_dict.keys())
        checkpoint_keys = set(checkpoint_state_dict.keys())
        overlap_keys = model_keys & checkpoint_keys
        missing_keys = list(incompatible_keys.missing_keys)
        unexpected_keys = list(incompatible_keys.unexpected_keys)
        logger.info(
            "Checkpoint load summary [%s]: model_keys=%d checkpoint_keys=%d overlap=%d missing=%d unexpected=%d",
            module_name,
            len(model_keys),
            len(checkpoint_keys),
            len(overlap_keys),
            len(missing_keys),
            len(unexpected_keys),
        )
        if missing_keys:
            logger.warning(
                "Checkpoint load missing keys [%s] (showing up to 10): %s",
                module_name,
                missing_keys[:10],
            )
        if unexpected_keys:
            logger.warning(
                "Checkpoint load unexpected keys [%s] (showing up to 10): %s",
                module_name,
                unexpected_keys[:10],
            )

    @staticmethod
    def _filter_compatible_state_dict(
        model_state_dict: dict[str, torch.Tensor],
        checkpoint_state_dict: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        filtered_state: dict[str, torch.Tensor] = {}
        skipped_shape_keys: list[str] = []
        for key, checkpoint_tensor in checkpoint_state_dict.items():
            model_tensor = model_state_dict.get(key, None)
            if model_tensor is None:
                continue
            if (
                not torch.is_tensor(checkpoint_tensor)
                or not torch.is_tensor(model_tensor)
                or tuple(checkpoint_tensor.shape) != tuple(model_tensor.shape)
            ):
                skipped_shape_keys.append(key)
                continue
            filtered_state[key] = checkpoint_tensor
        return filtered_state, skipped_shape_keys

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
