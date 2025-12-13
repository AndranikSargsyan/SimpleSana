import torch
from tqdm.auto import tqdm


def get_noise_predictor(model, prompt_embeds, neg_prompt_embeds, cfg_scale, model_kwargs=dict()):
    do_classifier_free_guidance = cfg_scale > 1.0

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
    
    def predict_noise(xt, timestep):
        xt_input = torch.cat([xt] * 2) if do_classifier_free_guidance else xt
        timestep = torch.full((xt_input.shape[0],), timestep, dtype=xt.dtype, device=xt.device)

        v_pred = model(
            xt_input,
            timestep,
            prompt_embeds,
            **model_kwargs,
        )
        
        if do_classifier_free_guidance:
            v_pred_uncond, v_pred_cond = v_pred.chunk(2)
            v_pred = v_pred_uncond + cfg_scale * (v_pred_cond - v_pred_uncond)

        return v_pred

    return predict_noise


def flow_euler(model, xt, timesteps):    
    for t_curr, t_prev in tqdm(list(zip(timesteps[:-1], timesteps[1:]))):
        v_pred = model(xt, t_curr)
        xt = xt + (t_prev - t_curr) * v_pred
    return xt
