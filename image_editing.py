from transformers import AutoConfig, AutoTokenizer
from qwen_vl_utils import smart_resize
import os
import torch
from PIL import Image
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
flux_decoder = FluxDecoder(flux_decoder_path, 't2i_models', device='cuda:0')

max_pixels = 262640
gen_size = 512

instruction = "<image> Make the car green." # <image> is a placeholder for the image
edit_images = ["assets/examples/car.png"]
instruction = instruction.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": instruction},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

images = [Image.open(image).convert('RGB') for image in edit_images]
# resize input to max_pixels to avoid oom
for j in range(len(images)):
    input_image = images[j]
    input_w, input_h = input_image.size
    resized_height, resized_width = smart_resize(
        input_h,
        input_w,
        max_pixels=max_pixels,
    )
    images[j] = input_image.resize((resized_width, resized_height))

inputs = processor(
    text=[text],
    images=images,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)
generation_image_grid_thw = torch.tensor([[1, 18, 18]]).to('cuda:0')
with torch.no_grad():
    outputs = model.generate(**inputs,
                             max_new_tokens=1024,
                             return_dict_in_generate=True,
                             generation_image_grid_thw=generation_image_grid_thw)
generated_ids = outputs['sequences']
output_image_embeddings = outputs['output_image_embeddings']

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
]
output_text = processor.batch_decode_all2all(
    generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
)
print(output_text)

pipe_kwargs = {"negative_prompt": "", "cfg_scale": 3.0}
image = flux_decoder.decode_image_embeds(output_image_embeddings, **pipe_kwargs, height=gen_size, width=gen_size)
image.save(f'editing.png')
