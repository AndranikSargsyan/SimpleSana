import torch
from mmcv import Registry
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as transformers_logging

from sana.model.dc_ae import DCAE_HF
from sana.model.utils import set_fp32_attention, set_grad_checkpoint

MODELS = Registry("models")

transformers_logging.set_verbosity_error()


def build_model(cfg, use_grad_checkpoint=False, use_fp32_attention=False, gc_step=1, **kwargs):
    if isinstance(cfg, str):
        cfg = dict(type=cfg)
    model = MODELS.build(cfg, default_args=kwargs)

    if use_grad_checkpoint:
        set_grad_checkpoint(model, gc_step=gc_step)
    if use_fp32_attention:
        set_fp32_attention(model)
    return model


def get_tokenizer_and_text_encoder(name, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained("Efficient-Large-Model/gemma-2-2b-it")
    tokenizer.padding_side = "right"
    text_encoder = (
        AutoModelForCausalLM.from_pretrained("Efficient-Large-Model/gemma-2-2b-it", torch_dtype=torch.bfloat16)
        .get_decoder()
        .to(device)
    )
    return tokenizer, text_encoder


def get_vae(model_path, device="cuda", dtype=None):
    dc_ae = DCAE_HF.from_pretrained(model_path).to(device).eval()
    return dc_ae.to(dtype)
