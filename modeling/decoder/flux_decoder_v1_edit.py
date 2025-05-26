import torch
from diffsynth import ModelManager
from diffsynth.models.utils import load_state_dict
from diffsynth.models.flux_dit import FluxDiT
from .flux_image_pipeline_v1_edit import FluxImagePipelineAll2All


class FluxDiTStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        return state_dict, {"disable_guidance_embedder": True}


def state_dict_converter():
    return FluxDiTStateDictConverter()


class FluxDecoder:

    def __init__(self, flux_all2all_modelpath, flux_path, device='cuda', torch_dtype=torch.bfloat16, enable_cpu_offload=False):
        self.device = device
        self.torch_dtype = torch_dtype
        self.enable_cpu_offload = enable_cpu_offload
        self.pipe, self.adapter, self.global_adapter = self.get_pipe(flux_all2all_modelpath, flux_path, device, torch_dtype)

    def get_pipe(self, flux_all2all_modelpath, flux_path, device="cuda", torch_dtype=torch.bfloat16):
        if self.enable_cpu_offload:
            model_manager = ModelManager(torch_dtype=torch_dtype, device='cpu')
        else:
            model_manager = ModelManager(torch_dtype=torch_dtype, device=device)
        model_manager.load_models([
            f"{flux_path}/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
            f"{flux_path}/FLUX/FLUX.1-dev/text_encoder_2",
            f"{flux_path}/FLUX/FLUX.1-dev/ae.safetensors",
        ])
        
        adapter = torch.nn.Sequential(torch.nn.Linear(3584, 4096),
                                      torch.nn.LayerNorm(4096), torch.nn.ReLU(),
                                      torch.nn.Linear(4096, 4096),
                                      torch.nn.LayerNorm(4096))
        global_adapter = torch.nn.Linear(3584, 768)

        state_dict = load_state_dict(flux_all2all_modelpath)

        adapter_state_dict = {key.replace("adapter.", ""): value for key, value in state_dict.items() if key.startswith('adapter.')}
        global_adapter_states = {key.replace("global_adapter.", ""): value for key, value in state_dict.items() if key.startswith('global_adapter.')}
        dit_state_dict = {key.replace("pipe.dit.", ""): value for key, value in state_dict.items() if not (key.startswith('global_adapter.') or key.startswith('adapter.'))}
        adapter.load_state_dict(adapter_state_dict)
        global_adapter.load_state_dict(global_adapter_states)
        adapter.to(device, dtype=torch_dtype)
        global_adapter.to(device, dtype=torch_dtype)


        FluxDiT.state_dict_converter = staticmethod(state_dict_converter)
        model_manager.load_model_from_single_file(flux_all2all_modelpath, state_dict=dit_state_dict, model_names=['flux_dit'], model_classes=[FluxDiT], model_resource='diffusers')

        pipe = FluxImagePipelineAll2All.from_model_manager(model_manager, device=device)
        if self.enable_cpu_offload:
            pipe.enable_cpu_offload()

        return pipe, adapter, global_adapter


    def preprocess_embeds(self, output_image_embeddings):
        if output_image_embeddings is None:
            return None
        if len(output_image_embeddings.size()) == 2:
            output_image_embeddings = output_image_embeddings.unsqueeze(0)
        output_image_embeddings = output_image_embeddings.to(device=self.device, dtype=self.torch_dtype)
        image_embed = self.adapter(output_image_embeddings)
        pooled_embed = self.global_adapter(output_image_embeddings.mean(dim=-2))
        prompt_emb = {}

        prompt_emb['prompt_emb'] = image_embed
        prompt_emb['pooled_prompt_emb'] = pooled_embed
        prompt_emb['text_ids'] = torch.zeros(image_embed.shape[0], image_embed.shape[1], 3).to(device=self.device, dtype=image_embed.dtype)
        return prompt_emb

    @torch.no_grad()
    def decode_image_embeds(self,
                            output_image_embeddings,
                            ref_image=None,
                            height=512,
                            width=512,
                            num_inference_steps=50,
                            seed=42,
                            negative_prompt="",
                            cfg_scale=1.0,
                            **pipe_kwargs):
        prompt_emb = self.preprocess_embeds(output_image_embeddings)
        image = self.pipe(prompt="",
                          prompt_emb=prompt_emb,
                          ref_image=ref_image,
                          num_inference_steps=num_inference_steps,
                          embedded_guidance=3.5,
                          negative_prompt=negative_prompt,
                          cfg_scale=cfg_scale,
                          height=height,
                          width=width,
                          seed=seed,
                          **pipe_kwargs)
        return image
