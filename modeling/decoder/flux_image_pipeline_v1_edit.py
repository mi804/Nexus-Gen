from typing import List
from tqdm import tqdm
import torch
from diffsynth.models import ModelManager
from diffsynth.controlnets import ControlNetConfigUnit
from diffsynth.prompters.flux_prompter import FluxPrompter
from diffsynth.pipelines.flux_image import FluxImagePipeline, TeaCache


class FluxPrompterAll2All(FluxPrompter):
    def encode_prompt(
        self,
        prompt,
        positive=True,
        device="cuda",
        t5_sequence_length=512,
        clip_only=False
    ):
        prompt = self.process_prompt(prompt, positive=positive)
        # CLIP
        pooled_prompt_emb = self.encode_prompt_using_clip(prompt, self.text_encoder_1, self.tokenizer_1, 77, device)
        if clip_only:
            return None, pooled_prompt_emb, None
        # T5
        prompt_emb = self.encode_prompt_using_t5(prompt, self.text_encoder_2, self.tokenizer_2, t5_sequence_length, device)
        # text_ids
        text_ids = torch.zeros(prompt_emb.shape[0], prompt_emb.shape[1], 3).to(device=device, dtype=prompt_emb.dtype)
        return prompt_emb, pooled_prompt_emb, text_ids


class FluxImagePipelineAll2All(FluxImagePipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompter = FluxPrompterAll2All()

    def encode_prompt(self, prompt, positive=True, t5_sequence_length=512, clip_only=False):
        prompt_emb, pooled_prompt_emb, text_ids = self.prompter.encode_prompt(
            prompt, device=self.device, positive=positive, t5_sequence_length=t5_sequence_length, clip_only=clip_only
        )
        return {"prompt_emb": prompt_emb, "pooled_prompt_emb": pooled_prompt_emb, "text_ids": text_ids}


    @staticmethod
    def from_model_manager(model_manager: ModelManager, controlnet_config_units: List[ControlNetConfigUnit]=[], prompt_refiner_classes=[], prompt_extender_classes=[], device=None, torch_dtype=None):
        pipe = FluxImagePipelineAll2All(
            device=model_manager.device if device is None else device,
            torch_dtype=model_manager.torch_dtype if torch_dtype is None else torch_dtype,
        )
        pipe.fetch_models(model_manager, controlnet_config_units, prompt_refiner_classes, prompt_extender_classes)
        return pipe


    def prepare_prompts(self, prompt, prompt_emb, prompt_emb_neg, local_prompts, masks, mask_scales, t5_sequence_length, negative_prompt, cfg_scale):
        # Extend prompt
        self.load_models_to_device(['text_encoder_1', 'text_encoder_2'])
        prompt, local_prompts, masks, mask_scales = self.extend_prompt(prompt, local_prompts, masks, mask_scales)

        # Encode prompts
        if prompt_emb is not None:
            prompt_emb_posi = prompt_emb
        else:
            prompt_emb_posi = self.encode_prompt(prompt, t5_sequence_length=t5_sequence_length)
        if prompt_emb_neg is not None:
            prompt_emb_nega = prompt_emb_neg
        else:        
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False, t5_sequence_length=t5_sequence_length) if cfg_scale != 1.0 else None
        prompt_emb_locals = [self.encode_prompt(prompt_local, t5_sequence_length=t5_sequence_length) for prompt_local in local_prompts]
        return prompt_emb_posi, prompt_emb_nega, prompt_emb_locals


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt,
        negative_prompt="",
        cfg_scale=1.0,
        embedded_guidance=3.5,
        t5_sequence_length=512,
        # Image
        input_image=None,
        denoising_strength=1.0,
        height=1024,
        width=1024,
        seed=None,
        # prompt_emb
        prompt_emb=None,
        ref_image=None,
        prompt_emb_neg=None,
        # Steps
        num_inference_steps=30,
        # local prompts
        local_prompts=(),
        masks=(),
        mask_scales=(),
        # ControlNet
        controlnet_image=None,
        controlnet_inpaint_mask=None,
        enable_controlnet_on_negative=False,
        # IP-Adapter
        ipadapter_images=None,
        ipadapter_scale=1.0,
        # EliGen
        eligen_entity_prompts=None,
        eligen_entity_masks=None,
        enable_eligen_on_negative=False,
        enable_eligen_inpaint=False,
        # TeaCache
        tea_cache_l1_thresh=None,
        # Tile
        tiled=False,
        tile_size=128,
        tile_stride=64,
        # Progress bar
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
        height, width = self.check_resize_height_width(height, width)

        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength)

        # Prepare latent tensors
        latents, input_latents = self.prepare_latents(input_image, height, width, seed, tiled, tile_size, tile_stride)

        # Prompt
        prompt_emb_posi, prompt_emb_nega, prompt_emb_locals = self.prepare_prompts(prompt, prompt_emb, prompt_emb_neg, local_prompts, masks, mask_scales, t5_sequence_length, negative_prompt, cfg_scale)

        # Extra input
        extra_input = self.prepare_extra_input(latents, guidance=embedded_guidance)

        # Entity control
        eligen_kwargs_posi, eligen_kwargs_nega, fg_mask, bg_mask = self.prepare_eligen(prompt_emb_nega, eligen_entity_prompts, eligen_entity_masks, width, height, t5_sequence_length, enable_eligen_inpaint, enable_eligen_on_negative, cfg_scale)

        # IP-Adapter
        ipadapter_kwargs_list_posi, ipadapter_kwargs_list_nega = self.prepare_ipadapter(ipadapter_images, ipadapter_scale)

        # ControlNets
        controlnet_kwargs_posi, controlnet_kwargs_nega, local_controlnet_kwargs = self.prepare_controlnet(controlnet_image, masks, controlnet_inpaint_mask, tiler_kwargs, enable_controlnet_on_negative)

        # TeaCache
        tea_cache_kwargs = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh) if tea_cache_l1_thresh is not None else None}


        if ref_image is not None:
            ref_image = self.preprocess_image(ref_image).to(device=self.device, dtype=self.torch_dtype)
            ref_latents = self.encode_image(ref_image)
            ref_image_ids = self.dit.prepare_image_ids(ref_latents)
            extra_input['image_ids'] = torch.concat([extra_input['image_ids'], ref_image_ids], dim=-2)

        # Denoise
        self.load_models_to_device(['dit', 'controlnet'])
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(self.device)
    
            all_latents = torch.concat([latents, ref_latents], dim=-2)
            # Positive side

            noise_pred = self.dit(all_latents, timestep=timestep, **prompt_emb, **extra_input)

            noise_pred = noise_pred[:, :, :latents.shape[2], :]
            # Iterate
            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)

            # UI
            if progress_bar_st is not None:
                progress_bar_st.progress(progress_id / len(self.scheduler.timesteps))

        # Decode image
        self.load_models_to_device(['vae_decoder'])
        image = self.decode_image(latents, **tiler_kwargs)

        # Offload all models
        self.load_models_to_device([])
        return image



def lets_dance_flux(
    dit,
    controlnet = None,
    step1x_connector = None,
    hidden_states=None,
    timestep=None,
    prompt_emb=None,
    pooled_prompt_emb=None,
    guidance=None,
    text_ids=None,
    image_ids=None,
    controlnet_frames=None,
    tiled=False,
    tile_size=128,
    tile_stride=64,
    entity_prompt_emb=None,
    entity_masks=None,
    ipadapter_kwargs_list={},
    id_emb=None,
    infinityou_guidance=None,
    flex_condition=None,
    flex_uncondition=None,
    flex_control_stop_timestep=None,
    step1x_llm_embedding=None,
    step1x_mask=None,
    step1x_reference_latents=None,
    tea_cache: TeaCache = None,
    **kwargs
):

    # ControlNet
    if controlnet is not None and controlnet_frames is not None:
        controlnet_extra_kwargs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "prompt_emb": prompt_emb,
            "pooled_prompt_emb": pooled_prompt_emb,
            "guidance": guidance,
            "text_ids": text_ids,
            "image_ids": image_ids,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }
        if id_emb is not None:
            controlnet_text_ids = torch.zeros(id_emb.shape[0], id_emb.shape[1], 3).to(device=hidden_states.device, dtype=hidden_states.dtype)
            controlnet_extra_kwargs.update({"prompt_emb": id_emb, 'text_ids': controlnet_text_ids, 'guidance': infinityou_guidance})
        controlnet_res_stack, controlnet_single_res_stack = controlnet(
            controlnet_frames, **controlnet_extra_kwargs
        )
        
    # Flex
    if flex_condition is not None:
        if timestep.tolist()[0] >= flex_control_stop_timestep:
            hidden_states = torch.concat([hidden_states, flex_condition], dim=1)
        else:
            hidden_states = torch.concat([hidden_states, flex_uncondition], dim=1)

    # Step1x
    if step1x_llm_embedding is not None:
        prompt_emb, pooled_prompt_emb = step1x_connector(step1x_llm_embedding, timestep / 1000, step1x_mask)
        text_ids = torch.zeros((1, prompt_emb.shape[1], 3), dtype=prompt_emb.dtype, device=prompt_emb.device)

    if image_ids is None:
        image_ids = dit.prepare_image_ids(hidden_states)

    conditioning = dit.time_embedder(timestep, hidden_states.dtype) + dit.pooled_text_embedder(pooled_prompt_emb)
    if dit.guidance_embedder is not None:
        guidance = guidance * 1000
        conditioning = conditioning + dit.guidance_embedder(guidance, hidden_states.dtype)

    height, width = hidden_states.shape[-2:]
    hidden_states = dit.patchify(hidden_states)

    # Step1x
    if step1x_reference_latents is not None:
        step1x_reference_image_ids = dit.prepare_image_ids(step1x_reference_latents)
        step1x_reference_latents = dit.patchify(step1x_reference_latents)
        image_ids = torch.concat([image_ids, step1x_reference_image_ids], dim=-2)
        hidden_states = torch.concat([hidden_states, step1x_reference_latents], dim=1)

    hidden_states = dit.x_embedder(hidden_states)

    if entity_prompt_emb is not None and entity_masks is not None:
        prompt_emb, image_rotary_emb, attention_mask = dit.process_entity_masks(hidden_states, prompt_emb, entity_prompt_emb, entity_masks, text_ids, image_ids)
    else:
        prompt_emb = dit.context_embedder(prompt_emb)
        image_rotary_emb = dit.pos_embedder(torch.cat((text_ids, image_ids), dim=1))
        attention_mask = None

    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, hidden_states, conditioning)
    else:
        tea_cache_update = False

    if tea_cache_update:
        hidden_states = tea_cache.update(hidden_states)
    else:
        # Joint Blocks
        for block_id, block in enumerate(dit.blocks):
            hidden_states, prompt_emb = block(
                hidden_states,
                prompt_emb,
                conditioning,
                image_rotary_emb,
                attention_mask,
                ipadapter_kwargs_list=ipadapter_kwargs_list.get(block_id, None)
            )
            # ControlNet
            if controlnet is not None and controlnet_frames is not None:
                hidden_states = hidden_states + controlnet_res_stack[block_id]

        # Single Blocks
        hidden_states = torch.cat([prompt_emb, hidden_states], dim=1)
        num_joint_blocks = len(dit.blocks)
        for block_id, block in enumerate(dit.single_blocks):
            hidden_states, prompt_emb = block(
                hidden_states,
                prompt_emb,
                conditioning,
                image_rotary_emb,
                attention_mask,
                ipadapter_kwargs_list=ipadapter_kwargs_list.get(block_id + num_joint_blocks, None)
            )
            # ControlNet
            if controlnet is not None and controlnet_frames is not None:
                hidden_states[:, prompt_emb.shape[1]:] = hidden_states[:, prompt_emb.shape[1]:] + controlnet_single_res_stack[block_id]
        hidden_states = hidden_states[:, prompt_emb.shape[1]:]

        if tea_cache is not None:
            tea_cache.store(hidden_states)

    hidden_states = dit.final_norm_out(hidden_states, conditioning)
    hidden_states = dit.final_proj_out(hidden_states)
    
    # Step1x
    if step1x_reference_latents is not None:
        hidden_states = hidden_states[:, :hidden_states.shape[1] // 2]

    hidden_states = dit.unpatchify(hidden_states, height, width)

    return hidden_states
