# Copyright (C) 2024 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file
# located at the root directory.

import torch
import gc

from OneActor.oa_diffusers import DDIMScheduler
from OneActor.consistory_unet_sdxl import ConsistorySDXLUNet2DConditionModel
from OneActor.consistory_pipeline import ConsistoryExtendAttnSDXLPipeline
from OneActor.consistory_utils import FeatureInjector, AnchorCache, StoryPipelineStore
from OneActor.utils.general_utils import *


from OneActor.utils.ptp_utils import view_images

LATENT_RESOLUTIONS = [32, 64]

def load_pipeline(gpu_id=0):
    float_type = torch.float16
    sd_id = "stabilityai/stable-diffusion-xl-base-1.0"
    
    device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() else torch.device('cpu')
    unet = ConsistorySDXLUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet", torch_dtype=float_type)
    scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler")

    story_pipeline = ConsistoryExtendAttnSDXLPipeline.from_pretrained(
        sd_id, unet=unet, torch_dtype=float_type, variant="fp16", use_safetensors=True, scheduler=scheduler
    ).to(device)
    story_pipeline.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
    
    return story_pipeline

def create_anchor_mapping(bsz, anchor_indices=[0]):
    anchor_mapping = torch.eye(bsz, dtype=torch.bool)
    for anchor_idx in anchor_indices:
        anchor_mapping[:, anchor_idx] = True

    return anchor_mapping

def create_token_indices(prompts, batch_size, concept_token, tokenizer):
    if isinstance(concept_token, str):
        concept_token = [concept_token]

    concept_token_id = [tokenizer.encode(x, add_special_tokens=False)[0] for x in concept_token]
    tokens = tokenizer.batch_encode_plus(prompts, padding=True, return_tensors='pt')['input_ids']

    token_indices = torch.full((len(concept_token), batch_size), -1, dtype=torch.int64)
    for i, token_id in enumerate(concept_token_id):
        batch_loc, token_loc = torch.where(tokens == token_id)
        token_indices[i, batch_loc] = token_loc

    return token_indices

def create_latents(story_pipeline, seed, batch_size, same_latent, device, float_type):
    # if seed is int
    if isinstance(seed, int):
        g = torch.Generator('cuda').manual_seed(seed)
        shape = (batch_size, story_pipeline.unet.config.in_channels, 128, 128)
        latents = randn_tensor(shape, generator=g, device=device, dtype=float_type)
    elif isinstance(seed, list):
        shape = (batch_size, story_pipeline.unet.config.in_channels, 128, 128)
        latents = torch.empty(shape, device=device, dtype=float_type)
        for i, seed_i in enumerate(seed):
            g = torch.Generator('cuda').manual_seed(seed_i)
            curr_latent = randn_tensor(shape, generator=g, device=device, dtype=float_type)
            latents[i] = curr_latent[i]

    if same_latent:
        latents = latents[:1].repeat(batch_size, 1, 1, 1)

    return latents, g

# Anchors
def run_anchor_generation(story_pipeline, prompts, concept_token,
                        seed=40, n_steps=30, mask_dropout=0.5,
                        same_latent=False, share_queries=True,
                        perform_sdsa=True, perform_injection=True,
                        downscale_rate=4, cache_cpu_offloading=False, story_pipeline_store=None):
    device = story_pipeline.device
    tokenizer = story_pipeline.tokenizer
    float_type = story_pipeline.dtype
    unet = story_pipeline.unet

    batch_size = len(prompts)

    token_indices = create_token_indices(prompts, batch_size, concept_token, tokenizer)

    default_attention_store_kwargs = {
        'token_indices': token_indices,
        'mask_dropout': mask_dropout
    }

    default_extended_attn_kwargs = {'extend_kv_unet_parts': ['up']}
    query_store_kwargs={'t_range': [0,n_steps//10], 'strength_start': 0.9, 'strength_end': 0.81836735}

    latents, g = create_latents(story_pipeline, seed, batch_size, same_latent, device, float_type)

    anchor_cache_first_stage = AnchorCache()
    anchor_cache_second_stage = AnchorCache()

    # ------------------ #
    # Extended attention First Run #

    if perform_sdsa:
        extended_attn_kwargs = {**default_extended_attn_kwargs, 't_range': [(1, n_steps)]}
    else:
        extended_attn_kwargs = {**default_extended_attn_kwargs, 't_range': []}

    print(extended_attn_kwargs['t_range'])
    out, prompt_embeds, xt_save, mid_save_list, mid_save_vanilla_list = story_pipeline(prompt=prompts, generator=g, latents=latents, 
                        attention_store_kwargs=default_attention_store_kwargs,
                        extended_attn_kwargs=extended_attn_kwargs,
                        share_queries=share_queries,
                        query_store_kwargs=query_store_kwargs,
                        anchors_cache=anchor_cache_first_stage,
                        num_inference_steps=n_steps)

    if story_pipeline_store is not None:
        story_pipeline_store.first_stage.images.append(out.images)
        story_pipeline_store.first_stage.prompt_embeds.append(prompt_embeds.cpu())
        story_pipeline_store.first_stage.xt_save.append([_.cpu() for _ in xt_save])
        story_pipeline_store.first_stage.mid_save_list.append([_.cpu() for _ in mid_save_list])
        story_pipeline_store.first_stage.prompt.append(prompts)
    
    img_all = view_images([np.array(x) for x in out.images], display_image=False, downscale_rate=downscale_rate)
    
    last_masks = story_pipeline.attention_store.last_mask

    # # dift_features = unet.latent_store.dift_features['261_0'][batch_size:]
    # dift_features = unet.latent_store.dift_features['265_0'][batch_size:]
    # dift_features = torch.stack([gaussian_smooth(x, kernel_size=3, sigma=1) for x in dift_features], dim=0)

    # anchor_cache_first_stage.dift_cache = dift_features
    # anchor_cache_first_stage.anchors_last_mask = last_masks

    # if cache_cpu_offloading:
    #     anchor_cache_first_stage.to_device(torch.device('cpu'))

    # nn_map, nn_distances = cyclic_nn_map(dift_features, last_masks, LATENT_RESOLUTIONS, device)

    if story_pipeline_store is not None:
        story_pipeline_store.first_stage.last_masks.append({key: last_masks[key].cpu() for key in last_masks})
        # story_pipeline_store.first_stage.nn_distances.append({key: nn_distances[key].cpu() for key in nn_distances})

    torch.cuda.empty_cache()
    gc.collect()

    return out.images, img_all, anchor_cache_first_stage
