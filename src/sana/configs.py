from dataclasses import dataclass, field
from typing import Optional

from sana.utils.config import ModelConfig, AEConfig, TextEncoderConfig, SchedulerConfig, SanaConfig


@dataclass
class SANA1_5_1600M_Config(SanaConfig):
    model: ModelConfig = field(default_factory=lambda: ModelConfig(
        model='SanaMS_1600M_P1_D20',
        teacher=None,
        image_size=1024,
        mixed_precision='bf16',
        fp32_attention=True,
        load_from='hf://Efficient-Large-Model/SANA1.5_1.6B_1024px/checkpoints/SANA1.5_1.6B_1024px.pth',
        discriminator_model=None,
        teacher_model=None,
        teacher_model_weight_dtype=None,
        resume_from={'checkpoint': None, 'load_ema': False, 'resume_lr_scheduler': True, 'resume_optimizer': True},
        aspect_ratio_type='ASPECT_RATIO_1024',
        multi_scale=True,
        pe_interpolation=1.0,
        micro_condition=False,
        attn_type='linear',
        autocast_linear_attn=False,
        ffn_type='glumbconv',
        mlp_acts=['silu', 'silu', None],
        mlp_ratio=2.5,
        use_pe=False,
        pos_embed_type='sincos',
        qk_norm=True,
        class_dropout_prob=0.1,
        linear_head_dim=32,
        cross_norm=True,
        cross_attn_type='flash',
        logvar=False,
        cfg_scale=4,
        cfg_embed=False,
        cfg_embed_scale=1.0,
        guidance_type='classifier-free',
        ladd_multi_scale=True,
        head_block_ids=None,
        extra=None
    ))

    vae: AEConfig = field(default_factory=lambda: AEConfig(
        vae_type='dc-ae',
        vae_pretrained='mit-han-lab/dc-ae-f32c32-sana-1.1',
        scale_factor=0.41407,
        vae_latent_dim=32,
        vae_downsample_rate=32,
        sample_posterior=True,
    ))

    text_encoder: TextEncoderConfig = field(default_factory=lambda: TextEncoderConfig(
        text_encoder_name='gemma-2-2b-it',
        caption_channels=2304,
        y_norm=True,
        y_norm_scale_factor=0.01,
        model_max_length=300,
        chi_prompt=[
            'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:',
            '- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.',
            '- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.',
            'Here are examples of how to transform or refine prompts:',
            '- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.',
            '- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.',
            'Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:',
            'User Prompt: '
        ],
        extra=None
    ))

    scheduler: SchedulerConfig = field(default_factory=lambda: SchedulerConfig(
        train_sampling_steps=1000,
        predict_flow_v=True,
        noise_schedule='linear_flow',
        pred_sigma=False,
        learn_sigma=True,
        vis_sampler='flow_euler',
        flow_shift=3.0,
        weighting_scheme='logit_normal',
        weighting_scheme_discriminator='logit_normal_trigflow',
        add_noise_timesteps=[1.5708],
        logit_mean=0.0,
        logit_std=1.0,
        logit_mean_discriminator=0.0,
        logit_std_discriminator=1.0,
        sigma_data=0.5,
        timestep_norm_scale_factor=1.0,
        extra=None
    ))

    debug: bool = False
    caching: bool = False
    name: str = 'baseline'
    loss_report_name: str = 'loss'
    image_size: int = 1024
    cfg_scale: float = 4.5
    seed: int = 42
    step: int = -1
    custom_image_size: Optional[int] = None
    shield_model_path: str = 'google/shieldgemma-2b'


@dataclass
class SANA1_5_4800M_Config(SANA1_5_1600M_Config):
    model: ModelConfig = field(default_factory=lambda: ModelConfig(
        model='SanaMS_4800M_P1_D60',
        teacher=None,
        image_size=1024,
        mixed_precision='bf16',
        fp32_attention=True,
        load_from='hf://Efficient-Large-Model/SANA1.5_4.8B_1024px/checkpoints/SANA1.5_4.8B_1024px.pth',
        discriminator_model=None,
        teacher_model=None,
        teacher_model_weight_dtype=None,
        resume_from={'checkpoint': None, 'load_ema': False, 'resume_lr_scheduler': True, 'resume_optimizer': True},
        aspect_ratio_type='ASPECT_RATIO_1024',
        multi_scale=True,
        pe_interpolation=1.0,
        micro_condition=False,
        attn_type='linear',
        autocast_linear_attn=False,
        ffn_type='glumbconv',
        mlp_acts=['silu', 'silu', None],
        mlp_ratio=2.5,
        use_pe=False,
        pos_embed_type='sincos',
        qk_norm=True,
        class_dropout_prob=0.1,
        linear_head_dim=32,
        cross_norm=True,
        cross_attn_type='flash',
        logvar=False,
        cfg_scale=4,
        cfg_embed=False,
        cfg_embed_scale=1.0,
        guidance_type='classifier-free',
        ladd_multi_scale=True,
        head_block_ids=None,
        extra=None
    ))


@dataclass
class SANA1_600M_Config(SANA1_5_1600M_Config):
    model: ModelConfig = field(default_factory=lambda: ModelConfig(
        model='SanaMS_600M_P1_D28',
        teacher=None,
        image_size=1024,
        mixed_precision='fp16',
        fp32_attention=True,
        load_from='hf://Efficient-Large-Model/Sana_600M_1024px/checkpoints/Sana_600M_1024px_MultiLing.pth',
        discriminator_model=None,
        teacher_model=None,
        teacher_model_weight_dtype=None,
        resume_from={'checkpoint': None, 'load_ema': False, 'resume_lr_scheduler': True, 'resume_optimizer': True},
        aspect_ratio_type='ASPECT_RATIO_1024',
        multi_scale=True,
        pe_interpolation=1.0,
        micro_condition=False,
        attn_type='linear',
        autocast_linear_attn=False,
        ffn_type='glumbconv',
        mlp_acts=['silu', 'silu', None],
        mlp_ratio=2.5,
        use_pe=False,
        pos_embed_type='sincos',
        qk_norm=False,
        class_dropout_prob=0.1,
        linear_head_dim=32,
        cross_attn_type='flash',
        logvar=False,
        cfg_scale=4,
        cfg_embed=False,
        cfg_embed_scale=1.0,
        guidance_type='classifier-free',
        ladd_multi_scale=True,
        head_block_ids=None,
        extra=None
    ))

    scheduler: SchedulerConfig = field(default_factory=lambda: SchedulerConfig(
        train_sampling_steps=1000,
        predict_flow_v=True,
        noise_schedule='linear_flow',
        pred_sigma=False,
        learn_sigma=True,
        vis_sampler='flow_euler',
        flow_shift=4.0,
        weighting_scheme='logit_normal',
        weighting_scheme_discriminator='logit_normal_trigflow',
        add_noise_timesteps=[1.5708],
        logit_mean=0.0,
        logit_std=1.0,
        logit_mean_discriminator=0.0,
        logit_std_discriminator=1.0,
        sigma_data=0.5,
        timestep_norm_scale_factor=1.0,
        extra=None
    ))
