import torch
from diffsynth import ModelManager
from diffsynth.models.utils import load_state_dict
from diffsynth.models.flux_dit import FluxDiT
from .pipelines import NexusGenGenerationPipeline


class FluxDiTStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        return state_dict


def state_dict_converter():
    return FluxDiTStateDictConverter()


class NexusGenGenerationDecoder:

    def __init__(self, flux_all2all_modelpath, flux_path, device='cuda', torch_dtype=torch.bfloat16, enable_cpu_offload=False):
        self.device = device
        self.torch_dtype = torch_dtype
        self.enable_cpu_offload = enable_cpu_offload
        self.pipe, self.adapter = self.get_pipe(flux_all2all_modelpath, flux_path, device, torch_dtype)

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

        state_dict = load_state_dict(flux_all2all_modelpath)
        adapter_state_dict = {key.replace("adapter.", ""): value for key, value in state_dict.items() if key.startswith('adapter.')}
        dit_state_dict = {key.replace("pipe.dit.", ""): value for key, value in state_dict.items() if not key.startswith('adapter.')}
        
        adapter = torch.nn.Sequential(torch.nn.Linear(3584, 4096),
                                      torch.nn.LayerNorm(4096), torch.nn.ReLU(),
                                      torch.nn.Linear(4096, 4096),
                                      torch.nn.LayerNorm(4096))
        adapter.load_state_dict(adapter_state_dict)
        adapter.to(device, dtype=torch_dtype)

        FluxDiT.state_dict_converter = staticmethod(state_dict_converter)
        model_manager.load_model_from_single_file(flux_all2all_modelpath, state_dict=dit_state_dict, model_names=['flux_dit'], model_classes=[FluxDiT], model_resource='diffusers')

        pipe = NexusGenGenerationPipeline.from_model_manager(model_manager, device=device)
        if self.enable_cpu_offload:
            pipe.enable_cpu_offload()

        return pipe, adapter

    @torch.no_grad()
    def decode_image_embeds(self,
                            output_image_embeddings,
                            num_inference_steps=50,
                            height=512,
                            width=512,
                            seed=42,
                            negative_prompt="",
                            cfg_scale=1.0,
                            embedded_guidance=3.5,
                            **pipe_kwargs):
        output_image_embeddings = output_image_embeddings.to(device=self.device, dtype=self.torch_dtype)
        image_embed = self.adapter(output_image_embeddings)
        image = self.pipe(prompt="",
                          image_embed=image_embed,
                          num_inference_steps=num_inference_steps,
                          embedded_guidance=embedded_guidance,
                          negative_prompt=negative_prompt,
                          cfg_scale=cfg_scale,
                          height=height,
                          width=width,
                          seed=seed,
                          **pipe_kwargs)
        return image
