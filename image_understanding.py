from transformers import AutoConfig
from qwen_vl_utils import process_vision_info
from modeling.ar.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from modeling.ar.processing_qwen2_5_vl import Qwen2_5_VLProcessor

model_path = 'models/Nexus-Gen'
model_config = AutoConfig.from_pretrained(model_path)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,
                                                           config=model_config,
                                                           trust_remote_code=True,
                                                           torch_dtype="auto",
                                                           device_map="cuda:0")
processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
model.eval()


messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "assets/examples/car.png",
            },
            {"type": "text", "text": "Describe the image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
image_inputs = [image.resize((512, 512)) for image in image_inputs]
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
