import gradio as gr
import torch
from PIL import Image
import os
from transformers import AutoConfig, AutoTokenizer
from qwen_vl_utils import process_vision_info, smart_resize
from modeling.decoder.flux_decoder import FluxDecoder
from modeling.ar.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from modeling.ar.processing_qwen2_5_vl import Qwen2_5_VLProcessor
import numpy as np

# Initialize model and processor
model_path = 'models/Nexus-Gen'
model_config = AutoConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    config=model_config,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="cuda:0"
)
processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
model.eval()

# Initialize Flux Decoder
flux_decoder_path = os.path.join(model_path, 'decoder_81_512.bin')
flux_decoder = FluxDecoder(flux_decoder_path, 'models', device='cuda:0')

# Define system prompt
SYSTEM_PROMPT = "You are Nexus-Gen, is a unified model that synergizes the language reasoning capabilities of LLMs with the image synthesis power of diffusion models."

def image_understanding(image, question):
    """Multimodal Q&A function - supports both visual and text Q&A"""
    if image is not None:
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": question if question else "Describe the image in detail."},
                ],
            }
        ]
    else:
        # Text-only Q&A mode
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                ],
            }
        ]
    
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    if image is not None:
        image_inputs, video_inputs = process_vision_info(messages)
        image_inputs = [img.resize((512, 512)) for img in image_inputs]
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    else:
        inputs = processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
    
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def image_generation(prompt):
    """Image generation function"""
    generation_instruction = 'Generate an image according to the following description: {}'
    prompt = generation_instruction.format(prompt)
    
    messages = [
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": prompt
            },],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generation_image_grid_thw = torch.tensor([[1, 18, 18]]).to('cuda:0')
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            return_dict_in_generate=True,
            generation_image_grid_thw=generation_image_grid_thw,
            use_cache=True
        )
        
        if not hasattr(outputs, 'output_image_embeddings'):
            image_embeddings = model.get_image_embeddings(outputs)
            if image_embeddings is not None:
                output_image_embeddings = image_embeddings
            else:
                raise ValueError("Failed to generate image embeddings")
        else:
            output_image_embeddings = outputs.output_image_embeddings
    
    pipe_kwargs = {"negative_prompt": "", "cfg_scale": 3.0}
    image = flux_decoder.decode_image_embeds(output_image_embeddings, **pipe_kwargs)
    return image

def image_generation_with_polish(prompt):
    """Enhanced image generation function with prompt optimization"""
    extend_prompt_instruction = '''Please enhance and enrich the following image generation prompt by adding more details, style and atmosphere descriptions to make it more specific and vivid: {}'''
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": extend_prompt_instruction.format(prompt)},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    extended_prompt = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    polished_image = image_generation(extended_prompt[0])
    return extended_prompt[0], polished_image

def image_editing(image, instruction):
    """Image editing function"""
    max_pixels = 262640
    gen_size = 512
    
    # Convert numpy array to PIL Image if needed
    input_image = Image.fromarray(image) if not isinstance(image, Image.Image) else image
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": input_image,
                },
                {"type": "text", "text": instruction},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Process input image
    input_w, input_h = input_image.size
    resized_height, resized_width = smart_resize(
        input_h,
        input_w,
        max_pixels=max_pixels,
    )
    resized_image = input_image.resize((resized_width, resized_height))
    
    inputs = processor(
        text=[text],
        images=[resized_image],
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
    
    output_image_embeddings = outputs['output_image_embeddings']
    pipe_kwargs = {"negative_prompt": "", "cfg_scale": 3.0}
    edited_image = flux_decoder.decode_image_embeds(output_image_embeddings, **pipe_kwargs, height=gen_size, width=gen_size)
    return edited_image

# Create Gradio interface
with gr.Blocks(title="Nexus-Gen Demo") as demo:
    gr.Markdown("# Nexus-Gen Demo")
    
    with gr.Tab("Multimodal Q&A"):
        with gr.Row():
            with gr.Column():
                understanding_input = gr.Image(label="Upload Image (Optional)")
                understanding_question = gr.Textbox(
                    label="Input Question", 
                    lines=2, 
                    placeholder="You can:\n1. Upload an image and ask questions about it\n2. Ask text-only questions\n3. Upload an image without a question for automatic description"
                )
                understanding_button = gr.Button("Generate Response")
            with gr.Column():
                understanding_output = gr.Markdown(label="Response")
        understanding_button.click(
            fn=image_understanding,
            inputs=[understanding_input, understanding_question],
            outputs=[understanding_output]
        )
        gr.Examples(
            examples=[
                # Visual Q&A examples
                ["assets/examples/car.png", "What color is this car?"],
                # Text Q&A examples
                [None, "What are the main differences between electric and traditional fuel vehicles?"],
                # Image description example
                ["assets/examples/car.png", ""],
            ],
            inputs=[understanding_input, understanding_question],
            outputs=understanding_output,
            fn=image_understanding,
            cache_examples=False,
        )
    
    with gr.Tab("Image Generation"):
        with gr.Row():
            with gr.Column():
                generation_input = gr.Textbox(label="Input Prompt", lines=3, placeholder="Describe the image you want to generate")
                use_polish = gr.Checkbox(label="Use Nexus-Gen to enhance prompt", value=False)
                generation_button = gr.Button("Generate Image")
            with gr.Column():
                polish_prompt = gr.Textbox(label="Enhanced Prompt", lines=4, interactive=False, visible=False)
                generation_output = gr.Image(label="Generated Result")
        
        def generate_with_option(prompt, use_polish):
            if use_polish:
                polished_prompt, image = image_generation_with_polish(prompt)
                return [polished_prompt, gr.update(visible=True), image]
            else:
                image = image_generation(prompt)
                return ["", gr.update(visible=False), image]
        
        use_polish.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[use_polish],
            outputs=[polish_prompt]
        )
        
        generation_button.click(
            fn=generate_with_option,
            inputs=[generation_input, use_polish],
            outputs=[polish_prompt, polish_prompt, generation_output]
        )
        
        gr.Examples(
            examples=[
                ["A beautiful sunset over a calm ocean, with palm trees silhouetted against the orange sky", False],
                ["A beautiful woman, sunrise", True],
            ],
            inputs=[generation_input, use_polish],
            outputs=[polish_prompt, polish_prompt, generation_output],
            fn=generate_with_option,
            cache_examples=False,
        )
    
    with gr.Tab("Image Editing"):
        with gr.Row():
            with gr.Column():
                editing_image = gr.Image(label="Upload Image to Edit")
                editing_instruction = gr.Textbox(label="Edit Instruction", lines=2)
                editing_button = gr.Button("Start Editing")
            with gr.Column():
                editing_output = gr.Image(label="Edited Result")
        editing_button.click(
            fn=image_editing,
            inputs=[editing_image, editing_instruction],
            outputs=[editing_output]
        )
        gr.Examples(
            examples=[
                ["assets/examples/car.png", "Make the car green"],
            ],
            inputs=[editing_image, editing_instruction],
            outputs=editing_output,
            fn=image_editing,
            cache_examples=False,
        )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )