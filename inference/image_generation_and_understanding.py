from transformers import AutoConfig, AutoTokenizer
import os
import torch
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from modeling.decoder.flux_decoder import FluxDecoder
from modeling.ar.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from modeling.ar.processing_qwen2_5_vl import Qwen2_5_VLProcessor


model_path = 'models/Nexus-Gen'
model_config = AutoConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,
                                                           config=model_config,
                                                           trust_remote_code=True,
                                                           torch_dtype="auto",
                                                           device_map="cuda:0")
processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
model.eval()

flux_decoder_path = os.path.join(model_path, 'decoder_81_512.bin') # path to trained decoder
flux_decoder = FluxDecoder(flux_decoder_path, 'models', device='cuda:0')

# image generation
prompt = "A middle-aged man with a graying beard and short hair stands on a quiet urban street, wearing a black jacket. He is looking off to the side with a thoughtful expression, his arms crossed. The background features blurred buildings with warm lights and a few indistinct figures walking in the distance. The scene has a calm, contemplative atmosphere."
generation_instruciton = 'Generate an image according to the following description: {}'
prompt = generation_instruciton.format(prompt)
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
pipe_kwargs = {"negative_prompt": "", "cfg_scale": 3.0}
image = flux_decoder.decode_image_embeds(output_image_embeddings, **pipe_kwargs)
image.save(f'generation.png')

assistance_message = [{
    "role": "assistant",
    "content": [{
        "type": "text",
        "text": output_text[0].replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
    },],
}]

messages += assistance_message

# understanding
messages += [{
    "role": "user",
    "content": [{
        "type": "text",
        "text": "How does the man's emotion appear?"
    },],
}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = processor(
    text=[text],
    images=[image],
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
