# Copyright (c) Alibaba, Inc. and its affiliates.
from swift.llm import Model, ModelGroup, ModelArch, ModelMeta, get_model_tokenizer_multimodal, register_model
from swift.llm.model.model.qwen import patch_qwen_vl_utils
from swift.llm.model.patcher import patch_output_clone, patch_output_to_input_device


def get_model_tokenizer_qwen2_5_all2all(*args, **kwargs):
    import os
    from modeling.ar.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

    height = int(os.environ.get('IMAGE_TRAIN_SIZE', 252))
    print(f'IMAGE_TRAIN_SIZE {height}')
    USE_DYNAMIC_RATIO = os.environ.get('USE_DYNAMIC_RATIO', 'False').lower() == 'true'
    print(f'USE_DYNAMIC_RATIO {USE_DYNAMIC_RATIO}')
    CONSISTANT_EDIT_SCALE = os.environ.get('CONSISTANT_EDIT_SCALE', 'False').lower() == 'true'
    print(f'CONSISTANT_EDIT_SCALE {CONSISTANT_EDIT_SCALE}')

    kwargs['automodel_class'] = Qwen2_5_VLForConditionalGeneration
    model, tokenizer = get_model_tokenizer_multimodal(*args, **kwargs)
    if model is not None and hasattr(model.model, 'embed_tokens'):
        patch_output_clone(model.model.embed_tokens)
        patch_output_to_input_device(model.model.embed_tokens)

    patch_qwen_vl_utils()
    return model, tokenizer


register_model(
    ModelMeta(
        model_type='qwenall2all_custom', 
        model_groups=[
            ModelGroup([
                Model(None, None, 'QwenAll2All'),
            ]),
        ],
        is_multimodal=True,
        model_arch=ModelArch.qwen2_vl,
        template='qwenall2all_custom',
        get_function=get_model_tokenizer_qwen2_5_all2all,
        architectures=['Qwen2_5_VLForConditionalGeneration'],
        requires=['transformers==4.49', 'qwen_vl_utils>=0.0.6', 'decord'],
        tags=['vision']))
