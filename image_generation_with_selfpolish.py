from transformers import AutoConfig
import os
import torch
from modeling.decoder.flux_decoder import FluxDecoder
from modeling.ar.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from modeling.ar.processing_qwen2_5_vl import Qwen2_5_VLProcessor


model_path = 'models/Nexus-Gen'
ckpt_path = 'output/v60-20250526-155430/checkpoint-4149'
model_config = AutoConfig.from_pretrained(ckpt_path)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(ckpt_path,
                                                           config=model_config,
                                                           trust_remote_code=True,
                                                           torch_dtype="auto",
                                                           device_map="cuda:0")
processor = Qwen2_5_VLProcessor.from_pretrained(ckpt_path)
model.eval()

prompt = "A beautiful woman, sunrise."
extend_prompt_instruction = 'Please extend and polish the following image generation prompt: {}'
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": extend_prompt_instruction.format(prompt)},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = processor(
    text=[text],
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
extended_prompt = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

generation_instruciton = 'Generate an image according to the following description: {}'
prompt = generation_instruciton.format(extended_prompt[0])
print(prompt)
messages = [{
    "role": "user",
    "content": [{
        "type": "text",
        "text": prompt
    },],
}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(
    text=[text],
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)
# [[1, 18, 18]] for 81 image tokens
generation_image_grid_thw = torch.tensor([[1, 18, 18]]).to('cuda:0')
with torch.no_grad():
    outputs = model.generate(**inputs,
                             max_new_tokens=1024,
                             return_dict_in_generate=True,
                             generation_image_grid_thw=generation_image_grid_thw)
generated_ids = outputs['sequences']
output_image_embeddings = outputs['output_image_embeddings']

generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)]
output_text = processor.batch_decode_all2all(generated_ids_trimmed,
                                                skip_special_tokens=False,
                                                clean_up_tokenization_spaces=False)
print(output_text)
model.cpu()

flux_decoder_path = os.path.join(model_path, 'decoder_81_512.bin') # path to trained decoder
flux_decoder = FluxDecoder(flux_decoder_path, 'models', device='cuda:0')

pipe_kwargs = {"negative_prompt": "", "cfg_scale": 3.0}
image = flux_decoder.decode_image_embeds(output_image_embeddings, **pipe_kwargs)
image.save(f'generation_selfpolish.png')
