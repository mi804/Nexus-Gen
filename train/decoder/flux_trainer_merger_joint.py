import lightning as pl
import torch, os
from train.decoder.messages_dataset import QwenVisual2Image
from modelscope.hub.api import HubApi
from diffsynth import ModelManager
from modeling.decoder.flux_image_pipeline import FluxImagePipelineAll2All
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup, AutoConfig
from modeling.decoder.modules import ImageEmbeddingMerger
from diffsynth.models.utils import load_state_dict
from modeling.ar.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from modeling.ar.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from PIL import Image
import numpy as np


class FluxForQwen(pl.LightningModule):
    def __init__(
        self,
        torch_dtype=torch.float16, pretrained_weights=[], preset_lora_path=None,
        learning_rate=1e-4, use_gradient_checkpointing=True, state_dict_converter=None,
        quantize = None, qwenvl_path=None, in_channel=3584, out_channel=4096, expand_ratio=4, lr_warmup_steps=500, load_from=None, num_layers=1,
    ):
        super().__init__()
        # Set parameters
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.state_dict_converter = state_dict_converter
        self.lora_alpha = None
        self.lr_warmup_steps = lr_warmup_steps
        # Load models
        model_manager = ModelManager(torch_dtype=torch_dtype, device=self.device)
        if quantize is None:
            model_manager.load_models(pretrained_weights)
        else:
            model_manager.load_models(pretrained_weights[1:])
            model_manager.load_model(pretrained_weights[0], torch_dtype=quantize)
        if preset_lora_path is not None:
            model_manager.load_lora(preset_lora_path)

        self.pipe = FluxImagePipelineAll2All.from_model_manager(model_manager)
        del self.pipe.vae_decoder

        self.embedding_merger = torch.nn.Sequential(
            torch.nn.Linear(in_channel, out_channel * expand_ratio),
            torch.nn.LayerNorm(out_channel * expand_ratio),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channel * expand_ratio, out_channel),
            torch.nn.LayerNorm(out_channel))

        self.pipe.scheduler.set_timesteps(1000, training=True)

        ckpt_path = '/mnt/nas1/zhanghong/project/all2all/modelscope/Nexus-Gen'
        model_config = AutoConfig.from_pretrained(ckpt_path)
        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(ckpt_path,
                                                                config=model_config,
                                                                trust_remote_code=True,
                                                                torch_dtype="auto",
                                                                device_map="cpu")
        self.qwen_processor = Qwen2_5_VLProcessor.from_pretrained(ckpt_path)

        self.freeze_parameters()


    def load_models(self):
        # This function is implemented in other modules
        self.pipe = None


    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.qwen_model.requires_grad_(False)
        self.qwen_model.eval()
        self.pipe.denoising_model().requires_grad_(True)
        self.pipe.denoising_model().train()
        self.embedding_merger.requires_grad_(True)
        self.embedding_merger.train()


    def get_target_embeddings(self, images, messages, num_img_tokens=81):
        images[-1] = images[-1].resize((252, 252))

        processor = self.qwen_processor
        model = self.qwen_model

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        text = text.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
        inputs = processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        input_embeds = model.model.embed_tokens(inputs['input_ids'])
        image_embeds = model.visual(inputs['pixel_values'], grid_thw=inputs['image_grid_thw'])
        ground_truth_image_embeds = image_embeds[-num_img_tokens:]
        input_image_embeds = image_embeds[:-num_img_tokens]

        image_mask = inputs['input_ids'] == model.config.image_token_id
        indices = image_mask.cumsum(dim=1)
        input_image_mask = torch.logical_and(indices <= (image_embeds.shape[0] - ground_truth_image_embeds.shape[0]), image_mask)
        gt_image_mask = torch.logical_and(image_mask, ~input_image_mask)
        input_image_mask = input_image_mask.unsqueeze(-1).expand_as(input_embeds)
        input_embeds = input_embeds.masked_scatter(input_image_mask, input_image_embeds)

        position_ids, _ = model.get_rope_index(inputs['input_ids'],
                                                    inputs['image_grid_thw'],
                                                    attention_mask=inputs['attention_mask'])
        position_ids = position_ids.contiguous()
        outputs = model(inputs_embeds=input_embeds, position_ids=position_ids, attention_mask=inputs['attention_mask'], return_dict=True)
        output_image_embeddings = outputs.image_embeddings[:, :-1, :]
        output_image_embeddings = output_image_embeddings[gt_image_mask[:, 1:]]
        output_image_embeddings = output_image_embeddings.unsqueeze(0)
        return output_image_embeddings


    def get_visual_embed(self, images):
        processor = self.qwen_processor
        visual_model = self.qwen_model.visual
        media_inputs = processor.image_processor(images=images, videos=None, return_tensors='pt', do_resize=False)
        pixel_values = media_inputs["pixel_values"].to(visual_model.device)
        image_grid_thw = media_inputs["image_grid_thw"].to(visual_model.device)
        pixel_values = pixel_values.type(visual_model.dtype)
        image_embeds = visual_model(pixel_values, grid_thw=image_grid_thw)
        return image_embeds


    def training_step(self, batch):
        # Data {"instruction": instruction, "image": image, "target_rgb_image": rgb_image, "ref_rgb_image": ref_rgb_image}

        messages, image, rgb_image, ref_rgb_image = batch["messages"], batch["image"], batch["target_rgb_image"], batch["ref_rgb_image"]
        image = image.unsqueeze(0)
        with torch.no_grad():
            target_embedding = self.get_target_embeddings([ref_rgb_image, rgb_image], messages)
            ref_embedding = self.get_visual_embed([ref_rgb_image.resize((504, 504))])

        # Prepare input parameters
        self.pipe.device = self.device
        prompt_emb = self.pipe.encode_prompt("", positive=True, clip_only=True)
        visual_emb = torch.cat([target_embedding, ref_embedding.unsqueeze(0)], dim=1)
        visual_emb = self.embedding_merger(visual_emb)
        prompt_emb['prompt_emb'] = visual_emb

        latents = self.pipe.vae_encoder(image.to(dtype=self.pipe.torch_dtype, device=self.device))

        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(self.device)
        extra_input = self.pipe.prepare_extra_input(latents, guidance=1.0)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # prepare embed_ids
        embed_text_ids = torch.zeros(target_embedding.shape[0], target_embedding.shape[1], 3).to(device=self.device, dtype=visual_emb.dtype)

        ref_embeds_grid = torch.tensor([[1, 36, 36]])
        batch_size, height, width = image.shape[0], ref_embeds_grid[0][1], ref_embeds_grid[0][2]
        ref_embed_ids = torch.zeros(height // 2, width // 2, 3)
        scale_factor_height, scale_factor_width = noisy_latents.shape[-2] / height, noisy_latents.shape[-1] / width
        ref_embed_ids[..., 1] = ref_embed_ids[..., 1] + torch.arange(height // 2)[:, None] * scale_factor_height.item()
        ref_embed_ids[..., 2] = ref_embed_ids[..., 2] + torch.arange(width // 2)[None, :] * scale_factor_width.item()
        ref_embed_ids = ref_embed_ids[None, :].repeat(batch_size, 1, 1, 1).reshape(batch_size, height // 2 * width // 2, 3)
        ref_embed_text_ids = ref_embed_ids.to(device=latents.device, dtype=latents.dtype)
        
        prompt_emb['text_ids'] = torch.cat([embed_text_ids, ref_embed_text_ids], dim=1)

        # Compute loss
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, **prompt_emb, **extra_input,
            use_gradient_checkpointing=self.use_gradient_checkpointing
        )
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        # Record log
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("train_learning_rate", lr, prog_bar=True, on_step=True, on_epoch=False, rank_zero_only=True)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False, rank_zero_only=True)
        return loss


    def configure_optimizers(self):
        import itertools
        trainable_modules = itertools.chain(
            self.embedding_merger.parameters(),
            self.pipe.denoising_model().parameters()
        )
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
    
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.lr_warmup_steps
        print('total_steps:', total_steps)

        # scheduler = get_cosine_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=warmup_steps,
        #     num_training_steps=total_steps,
        # )
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        dit_state_dict = self.pipe.denoising_model().state_dict()
        dit_state_dict = {f"pipe.dit.{key}": value for key, value in dit_state_dict.items()}
        checkpoint.update(dit_state_dict)

        merger_state_dict = self.embedding_merger.state_dict()
        merger_state_dict = {f"embedding_merger.{key}": value for key, value in merger_state_dict.items()}
        checkpoint.update(merger_state_dict)



def add_general_parsers(parser):
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=False,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width.",
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="Whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        choices=["32", "16", "16-mixed", "bf16"],
        help="Training precision",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=1,
        help="lr_warmup_steps.",
    )
    parser.add_argument(
        "--modelscope_model_id",
        type=str,
        default=None,
        help="Model ID on ModelScope (https://www.modelscope.cn/). The model will be uploaded to ModelScope automatically if you provide a Model ID.",
    )
    parser.add_argument(
        "--modelscope_access_token",
        type=str,
        default=None,
        help="Access key on ModelScope (https://www.modelscope.cn/). Required if you want to upload the model to ModelScope.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--use_wandb",
        default=True,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    return parser


def launch_training_task(model, args):
    print(args)
    # dataset and data loader
    dataset = QwenVisual2Image(
        args.dataset_path,
        steps_per_epoch=args.steps_per_epoch,
        height=args.height,
        width=args.width,
        center_crop=args.center_crop,
        random_flip=args.random_flip
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=lambda x: x[0]
    )
    # train
    if args.use_wandb:        
        from pytorch_lightning.loggers import WandbLogger

        wandb_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
        wandb_config.update(vars(args))
        import os
        os.makedirs(os.path.join(args.output_path, "wandb"), exist_ok=True)
        wandb_logger = WandbLogger(
            project="diffsynth_studio",
            name="diffsynth_studio",
            config=wandb_config,
            save_dir=os.path.join(args.output_path, "wandb")
        )
        
        logger = wandb_logger
        print("Using WandbLogger")
    else:
        logger = None
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision=args.precision,
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1)],
        logger=logger,
        log_every_n_steps=5,
    )
    trainer.fit(model=model, train_dataloaders=train_loader)

    # Upload models
    if args.modelscope_model_id is not None and args.modelscope_access_token is not None:
        print(f"Uploading models to modelscope. model_id: {args.modelscope_model_id} local_path: {trainer.log_dir}")
        with open(os.path.join(trainer.log_dir, "configuration.json"), "w", encoding="utf-8") as f:
            f.write('{"framework":"Pytorch","task":"text-to-image-synthesis"}\n')
        api = HubApi()
        api.login(args.modelscope_access_token)
        api.push_model(model_id=args.modelscope_model_id, model_dir=trainer.log_dir)
