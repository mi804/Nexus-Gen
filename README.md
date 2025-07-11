<p align="center">
    <br>
    <img src="assets/logo.jpg"/>
    <br>
<p>
<h1 align="center">Nexus-Gen: A Unified Model for Image Understanding, Generation, and Editing</h1>
 
<div align="center">

  <a href="http://arxiv.org/abs/2504.21356"><img src="https://img.shields.io/static/v1?label=Tech%20Report&message=Arxiv&color=red"></a> &ensp;
  <a href="https://www.modelscope.cn/models/DiffSynth-Studio/Nexus-Gen"><img src="https://img.shields.io/static/v1?label=Model&message=ModelScope&color=blue"></a> &ensp;
  <a href="https://huggingface.co/modelscope/Nexus-Gen"><img src="https://img.shields.io/static/v1?label=Model&message=HuggingFace&color=yellow"></a> &ensp;
  <a href="https://www.modelscope.cn/studios/DiffSynth-Studio/Nexus-Gen"><img src="https://img.shields.io/static/v1?label=Online%20Demo&message=ModeScope&color=green"></a> &ensp;

</div>

## News
- **July 11, 2025**: [Nexus-Gen V2](https://www.modelscope.cn/models/DiffSynth-Studio/Nexus-GenV2) is released, which is opitimized from the following aspects:
  - Better image understanding capbility (45.7 on [MMMU](https://github.com/MMMU-Benchmark/MMMU)) through optimization on training schedules.
  - Better image generation (0.81 on [GenEval](https://github.com/djghosh13/geneval.git)) robustness through training with long-short caption.
  - Better reconstruction in image editing tasks. We have proposed a better editing decoder for Nexus-Gen.
  - Support generation and editing with Chinese prompts.
- **May 27, 2025**: We fine-tuned Nexus-Gen using the [BLIP-3o-60k](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k) dataset, significantly improving the model's robustness to text prompts in image generation, **achieving a GenEval score of 0.79**. The [model checkpoints](https://www.modelscope.cn/models/DiffSynth-Studio/Nexus-Gen) have been updated.

## What is Nexus-Gen
Nexus-Gen is a unified model that synergizes the language reasoning capabilities of LLMs with the image synthesis power of diffusion models. We propose a unified image embedding spaces to model image understanding, generation and editing tasks. To jointly optimize Nexus-Gen across these tasks, we propose a multi-stage training strategy, which perform multitask pretraining on the autoregressive model and conditional adaption on the vision decoders.
ing, generation and editing tasks.

![architecture](assets/illustrations/architecture.jpg)


## Getting Started
### Installation
```shell
# 1. Install [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio.git) from source
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .

# 2. Install requirements
pip install -r requirements.txt

# 3. Install ms-swift if you want to perform finetuning on Nexus-Gen.
pip install ms-swift==3.3.0.dev0
```

### Prepare models
Nexus-Gen adopts Qwen2.5-VL-Instruct 7B as its autoregressive model, and adopts FLUX.1-Dev as the vision decoders (including the generation decoder and editing decoder). You can run the following scripts to download the checkpoints.
```shell
python download_models.py
```
### Image Understanding
Nexus-Gen inheret the image understanding ability of Qwen2.5-VL. Try the following script (Needs at least 17 GB VRAM).
```shell
python image_understanding.py --input_image assets/examples/cat.png --instruction "Please give a brief description of the image"
```

### Image Generation
Try the following scripts to perform image generation (Needs at least 24 GB VRAM). Please see `image_generation.py` for details about the inference hyperparameters.
```shell
python image_generation.py --prompt "A cute cat" --width 512 --height 512
```
Nexus-GenV2 supports generation with chinese prompts. You may further set the Chinese template for image generation by setting `--language zh` as follows.
```shell
python image_generation.py --prompt "一只可爱的猫" --language zh --width 1024 --height 1024
```
### Image Editing
The Nexus-Gen model comprises two decoders: a generation decoder and an editing decoder (recommended). The former directly utilizes the 81-dimensional embeddings output by the autoregressive model to generate images, while the latter additionally incorporates the original image's 324-dimensional embeddings, enabling more accurate reconstruction of unedited regions in the image.

Try the follow script to perform image editing with editing decoder.
```shell
python image_editing.py --input_image assets/examples/cat.png  --instruction "Add a pair of sunglasses"
```

When performing large-region image edits such as conceptual modifications, it is recommended to employ the generation decoder. This approach allows the model's image generation capabilities to directly enhance its editing performance. Try the follow script to perform image editing with generation decoder.
```shell
python image_editing.py --input_image assets/examples/cat.png  --instruction "The cat is now running in a forest." --use_generation_decoder
```

Nexus-Gen also supports image editing using Chinese prompts:
```shell
python image_editing.py --input_image assets/examples/cat.png  --instruction "给猫加一副太阳镜"
```

### Gradio demo
```shell
python app.py
```

## Training
Nexus-Gen is trained base on [ms-swift](https://github.com/modelscope/ms-swift.git) and [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio.git). You can find the training scripts in `train/scripts/train_decoder.sh` and `train_llm.sh`.

## Qualitative results of Nexus-Gen
![cover](assets/illustrations/gen_edit.jpg)

### Citation
```
@misc{zhang2025nexusgenunifiedmodelimage,
      title={Nexus-Gen: A Unified Model for Image Understanding, Generation, and Editing}, 
      author={Hong Zhang and Zhongjie Duan and Xingjun Wang and Yuze Zhao and Weiyi Lu and Zhipeng Di and Yixuan Xu and Yingda Chen and Yu Zhang},
      year={2025},
      eprint={2504.21356},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.21356v2}, 
}
```