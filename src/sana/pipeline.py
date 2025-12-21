import warnings
from typing import Optional, Callable

import torch
import torch.nn as nn
from PIL import Image

warnings.filterwarnings("ignore")  # ignore warning


from sana.configs import SANA1_5_1600M_Config
from sana.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae
from sana.model.utils import get_weight_dtype, resize_and_crop_tensor
from sana.scheduler import get_schedule
from sana.samplers import get_noise_predictor, flow_euler 
from sana.utils.download import find_model
from sana.utils.aspect import ASPECT_RATIO_1024_TEST, classify_height_width_bin
from sana.utils.config import SanaConfig, model_init_config


class SanaPipeline(nn.Module):
    def __init__(
        self,
        config: Optional[SanaConfig] = SANA1_5_1600M_Config(),
        sampler: Callable = flow_euler,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        self.args = self.config = config
        self.sampler = sampler

        # set some hyper-parameters
        self.image_size = self.config.model.image_size

        self.device = device

        self.latent_size = self.image_size // config.vae.vae_downsample_rate
        self.max_sequence_length = config.text_encoder.model_max_length
        self.flow_shift = config.scheduler.flow_shift

        weight_dtype = get_weight_dtype(config.model.mixed_precision)
        self.weight_dtype = weight_dtype
        self.vae_dtype = get_weight_dtype(config.vae.weight_dtype)

        self.base_ratios = ASPECT_RATIO_1024_TEST

        # 1. build vae and text encoder
        self.vae = self.build_vae(config.vae)
        self.tokenizer, self.text_encoder = self.build_text_encoder(config.text_encoder)

        # 2. build Sana model
        self.model = self.build_sana_model(config).to(self.device)

    def set_sampler(self, sampler):
        self.sampler = sampler

    def build_vae(self, config):
        vae = get_vae(config.vae_pretrained, self.device).to(self.vae_dtype)
        return vae

    def build_text_encoder(self, config):
        tokenizer, text_encoder = get_tokenizer_and_text_encoder(name=config.text_encoder_name, device=self.device)
        return tokenizer, text_encoder

    def build_sana_model(self, config):
        # model setting
        model_kwargs = model_init_config(config, latent_size=self.latent_size)
        model = build_model(
            config.model.model,
            use_fp32_attention=config.model.get("fp32_attention", False) and config.model.mixed_precision != "bf16",
            **model_kwargs,
        )
        return model

    def from_pretrained(self, model_path=None):
        if model_path is None:
            model_path = self.config.model.load_from

        state_dict = find_model(model_path)
        state_dict = state_dict.get("state_dict", state_dict)
        if "pos_embed" in state_dict:
            del state_dict["pos_embed"]
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        self.model.eval().to(self.weight_dtype)

    @torch.inference_mode()
    def vae_encode(self, images):
        scaling_factor = self.vae.cfg.scaling_factor if self.vae.cfg.scaling_factor is not None else 0.41407
        z = self.vae.encode(images.to(self.device))
        z = z * scaling_factor
        return z

    @torch.inference_mode()
    def vae_decode(self, latent):
        vae_scale_factor = (
            2 ** (len(self.vae.config.encoder_block_out_channels) - 1)
            if hasattr(self.vae, "config") and self.vae.config is not None
            else 32
        )
        scaling_factor = self.vae.cfg.scaling_factor if self.vae.cfg.scaling_factor else 0.41407
        if latent.shape[-1] * vae_scale_factor > 4000 or latent.shape[-2] * vae_scale_factor > 4000:
            from patch_conv import convert_model
            self.vae = convert_model(self.vae, splits=4)
        samples = self.vae.decode(latent.detach() / scaling_factor)
        return samples

    @torch.inference_mode()
    def forward(
        self,
        prompt=None,
        height=1024,
        width=1024,
        negative_prompt="",
        num_inference_steps=20,
        guidance_scale=4.5,
        generator=torch.Generator().manual_seed(42),
        latents=None,
        start_step=1.0,
        use_resolution_binning=True,
        use_chi_prompt=True,
        shift_schedule=True,
    ):
        self.ori_height, self.ori_width = height, width
        if use_resolution_binning:
            self.height, self.width = classify_height_width_bin(height, width, ratios=self.base_ratios)
        else:
            self.height, self.width = height, width
        self.latent_size_h, self.latent_size_w = (
            self.height // self.config.vae.vae_downsample_rate,
            self.width // self.config.vae.vae_downsample_rate,
        )
        
        if prompt is None:
            prompt = ""

        # data prepare
        hw, ar = (
            torch.tensor([[self.image_size, self.image_size]], dtype=torch.float, device=self.device),
            torch.tensor([[1.0]], device=self.device),
        )

        with torch.no_grad():
            # prepare text feature
            if not self.config.text_encoder.chi_prompt or not use_chi_prompt:
                max_length_all = self.config.text_encoder.model_max_length
                prompts_all = prompt
            else:
                chi_prompt = "\n".join(self.config.text_encoder.chi_prompt)
                prompts_all = chi_prompt + prompt
                num_chi_prompt_tokens = len(self.tokenizer.encode(chi_prompt))
                max_length_all = (
                    num_chi_prompt_tokens + self.config.text_encoder.model_max_length - 2
                )  # magic number 2: [bos], [_]

            # prepare prompt embeds
            prompt_token = self.tokenizer(
                prompts_all,
                max_length=max_length_all,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device=self.device)
            select_index = [0] + list(range(-self.config.text_encoder.model_max_length + 1, 0))
            prompt_embeds = self.text_encoder(prompt_token.input_ids, prompt_token.attention_mask)[0][:, None][
                :, :, select_index
            ].to(self.weight_dtype)
            prompt_embed_masks = prompt_token.attention_mask[:, select_index]

            # prepare negative prompt embeds
            neg_prompt_token = self.tokenizer(
                negative_prompt,
                max_length=self.max_sequence_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            neg_prompt_embeds = self.text_encoder(neg_prompt_token.input_ids, neg_prompt_token.attention_mask)[0]
            neg_prompt_embeds = neg_prompt_embeds[:, None].to(self.weight_dtype)

            # prepare latents
            if latents is None:
                z = torch.randn(
                    1,
                    self.config.vae.vae_latent_dim,
                    self.latent_size_h,
                    self.latent_size_w,
                    generator=generator,
                    device=self.device,
                    # dtype=self.weight_dtype,
                )
            else:
                z = latents.to(self.device)
            
            # prepare model kwargs
            model_kwargs = dict(data_info={"img_hw": hw, "aspect_ratio": ar}, mask=prompt_embed_masks)

            # prepare noise predictor
            noise_predictor = get_noise_predictor(
                model=self.model,
                prompt_embeds=prompt_embeds,
                neg_prompt_embeds=neg_prompt_embeds,
                cfg_scale=guidance_scale,
                model_kwargs=model_kwargs,
            )

            timesteps = get_schedule(num_inference_steps, z.shape[2], shift=shift_schedule)
            timesteps = [t for t in timesteps if t <= start_step]
            
            sample = self.sampler(
                model=noise_predictor,
                xt=z,
                timesteps=timesteps,
            )

        sample = sample.to(self.vae_dtype)
        with torch.no_grad():
            sample = self.vae_decode(sample)

        if use_resolution_binning:
            sample = resize_and_crop_tensor(sample, self.ori_width, self.ori_height)

        sample = Image.fromarray(sample[0].clone().clamp_(min=-1, max=1).add_(1).div_(2).mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy())

        return sample
