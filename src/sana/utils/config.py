import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class BaseConfig:
    def get(self, attribute_name, default=None):
        return getattr(self, attribute_name, default)

    def pop(self, attribute_name, default=None):
        if hasattr(self, attribute_name):
            value = getattr(self, attribute_name)
            delattr(self, attribute_name)
            return value
        else:
            return default

    def __str__(self):
        return json.dumps(asdict(self), indent=4)


@dataclass
class ModelConfig(BaseConfig):
    model: str = "SanaMS_600M_P1_D28"
    teacher: Optional[str] = None
    image_size: int = 512
    mixed_precision: str = "fp16"  # ['fp16', 'fp32', 'bf16']
    fp32_attention: bool = True
    load_from: Optional[str] = None
    discriminator_model: Optional[str] = None
    teacher_model: Optional[str] = None
    teacher_model_weight_dtype: Optional[str] = None
    resume_from: Optional[Union[Dict[str, Any], str]] = field(
        default_factory=lambda: {
            "checkpoint": None,
            "load_ema": False,
            "resume_lr_scheduler": True,
            "resume_optimizer": True,
        }
    )
    aspect_ratio_type: str = "ASPECT_RATIO_1024"
    multi_scale: bool = True
    pe_interpolation: float = 1.0
    micro_condition: bool = False
    attn_type: str = "linear"
    autocast_linear_attn: bool = False
    ffn_type: str = "glumbconv"
    mlp_acts: List[Optional[str]] = field(default_factory=lambda: ["silu", "silu", None])
    mlp_ratio: float = 2.5
    use_pe: bool = False
    pos_embed_type: str = "sincos"
    qk_norm: bool = False
    class_dropout_prob: float = 0.0
    linear_head_dim: int = 32
    cross_norm: bool = False
    cross_attn_type: str = "flash"
    logvar: bool = False
    cfg_scale: int = 4
    cfg_embed: bool = False
    cfg_embed_scale: float = 1.0
    guidance_type: str = "classifier-free"
    # for ladd
    ladd_multi_scale: bool = True
    head_block_ids: Optional[List[int]] = None
    extra: Any = None


@dataclass
class AEConfig(BaseConfig):
    vae_type: str = "AutoencoderDC"
    vae_pretrained: str = "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers"
    weight_dtype: str = "float32"
    scale_factor: float = 0.41407
    vae_latent_dim: int = 32
    vae_downsample_rate: int = 32
    sample_posterior: bool = True
    extra: Any = None


@dataclass
class TextEncoderConfig(BaseConfig):
    text_encoder_name: str = "gemma-2-2b-it"
    caption_channels: int = 2304
    y_norm: bool = True
    y_norm_scale_factor: float = 1.0
    model_max_length: int = 300
    chi_prompt: List[Optional[str]] = field(default_factory=lambda: [])
    extra: Any = None


@dataclass
class SchedulerConfig(BaseConfig):
    train_sampling_steps: int = 1000
    predict_flow_v: bool = True
    noise_schedule: str = "linear_flow"
    pred_sigma: bool = False
    learn_sigma: bool = True
    vis_sampler: str = "flow_dpm-solver"
    flow_shift: float = 1.0
    # logit-normal timestep
    weighting_scheme: Optional[str] = "logit_normal"
    weighting_scheme_discriminator: Optional[str] = "logit_normal_trigflow"
    add_noise_timesteps: List[float] = field(default_factory=lambda: [1.57080])
    logit_mean: float = 0.0
    logit_std: float = 1.0
    logit_mean_discriminator: float = 0.0
    logit_std_discriminator: float = 1.0
    sigma_data: float = 0.5
    timestep_norm_scale_factor: float = 1.0
    extra: Any = None


@dataclass
class ControlNetConfig(BaseConfig):
    control_signal_type: str = "scribble"
    validation_scribble_maps: List[str] = field(
        default_factory=lambda: [
            "output/tmp_embed/controlnet/dog_scribble_thickness_3.jpg",
            "output/tmp_embed/controlnet/girl_scribble_thickness_3.jpg",
            "output/tmp_embed/controlnet/cyborg_scribble_thickness_3.jpg",
            "output/tmp_embed/controlnet/Astronaut_scribble_thickness_3.jpg",
            "output/tmp_embed/controlnet/mountain_scribble_thickness_3.jpg",
        ]
    )


@dataclass
class ModelGrowthConfig(BaseConfig):
    """Model growth configuration for initializing larger models from smaller ones"""

    pretrained_ckpt_path: str = ""
    init_strategy: str = "constant"  # ['cyclic', 'block_expand', 'progressive', 'interpolation', 'random', 'constant']
    init_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "expand_ratio": 3,
            "noise_scale": 0.01,
        }
    )
    source_num_layers: int = 20
    target_num_layers: int = 60


@dataclass
class SanaConfig(BaseConfig):
    model: ModelConfig
    vae: AEConfig
    text_encoder: TextEncoderConfig
    scheduler: SchedulerConfig
    controlnet: Optional[ControlNetConfig] = None
    model_growth: Optional[ModelGrowthConfig] = None
    work_dir: str = "output/"
    resume_from: Optional[str] = None
    load_from: Optional[str] = None
    debug: bool = False
    caching: bool = False
    report_to: str = "wandb"
    tracker_project_name: str = "sana-baseline"
    name: str = "baseline"
    loss_report_name: str = "loss"


def model_init_config(config: SanaConfig, latent_size: int = 32):

    pred_sigma = getattr(config.scheduler, "pred_sigma", True)
    learn_sigma = getattr(config.scheduler, "learn_sigma", True) and pred_sigma
    return {
        "input_size": latent_size,
        "pe_interpolation": config.model.pe_interpolation,
        "config": config,
        "model_max_length": config.text_encoder.model_max_length,
        "qk_norm": config.model.qk_norm,
        "micro_condition": config.model.micro_condition,
        "caption_channels": config.text_encoder.caption_channels,
        "class_dropout_prob": config.model.class_dropout_prob,
        "y_norm": config.text_encoder.y_norm,
        "attn_type": config.model.attn_type,
        "ffn_type": config.model.ffn_type,
        "mlp_ratio": config.model.mlp_ratio,
        "mlp_acts": list(config.model.mlp_acts),
        "in_channels": config.vae.vae_latent_dim,
        "y_norm_scale_factor": config.text_encoder.y_norm_scale_factor,
        "use_pe": config.model.use_pe,
        "pos_embed_type": config.model.pos_embed_type,
        "linear_head_dim": config.model.linear_head_dim,
        "pred_sigma": pred_sigma,
        "learn_sigma": learn_sigma,
        "cross_norm": config.model.cross_norm,
        "cross_attn_type": config.model.cross_attn_type,
        "timestep_norm_scale_factor": config.scheduler.timestep_norm_scale_factor,
    }
