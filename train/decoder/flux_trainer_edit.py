import lightning as pl
import torch, os
from train.decoder.embed_image_dataset_edit import QwenVisual2Image
from modelscope.hub.api import HubApi
from diffsynth import ModelManager, FluxImagePipeline
from modeling.decoder.flux_image_pipeline import FluxImagePipelineAll2All
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup



class FluxForQwen(pl.LightningModule):
    def __init__(
        self,
        torch_dtype=torch.float16, pretrained_weights=[], preset_lora_path=None,
        learning_rate=1e-4, use_gradient_checkpointing=True, state_dict_converter=None,
        quantize = None, in_channel=3584, out_channel=4096, global_out_channel=768, expand_ratio=1, lr_warmup_steps=500, load_from=None,
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

        if quantize is not None:
            self.pipe.dit.quantize()

        self.pipe.scheduler.set_timesteps(1000, training=True)

        self.adapter = torch.nn.Sequential(
            torch.nn.Linear(in_channel, out_channel * expand_ratio),
            torch.nn.LayerNorm(out_channel * expand_ratio),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channel * expand_ratio, out_channel),
            torch.nn.LayerNorm(out_channel))

        self.global_adapter = torch.nn.Linear(in_channel, global_out_channel)

        if load_from is not None:
            print(f"Loading from {load_from}")
            state_dict = torch.load(load_from, weights_only=True, map_location="cpu")
            adapter_states = ['0.weight', '0.bias', '1.weight', '1.bias', '3.weight', '3.bias', '4.weight', '4.bias']
            adapter_state_dict = {}
            dit_state_dict = {}

            for key, value in state_dict.items():
                if key in adapter_states:
                    adapter_state_dict[key] = state_dict[key]
                else:
                    dit_state_dict[key] = state_dict[key]
            self.adapter.load_state_dict(adapter_state_dict)
            self.pipe.denoising_model().load_state_dict(dit_state_dict)

        self.freeze_parameters()


    def load_models(self):
        # This function is implemented in other modules
        self.pipe = None


    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().requires_grad_(True)
        self.pipe.denoising_model().train()
        self.adapter.requires_grad_(True)
        self.adapter.train()
        self.global_adapter.requires_grad_(True)
        self.global_adapter.train()


    def training_step(self, batch, batch_idx):
        # Data
        embed, image, ref_image = batch["embed"], batch["image"], batch["ref_image"]

        # Prepare input parameters
        self.pipe.device = self.device
        pooled_embed = self.global_adapter(embed.mean(dim=1))
        visual_emb = self.adapter(embed)
        prompt_emb = {}
        prompt_emb['prompt_emb'] = visual_emb
        prompt_emb['pooled_prompt_emb'] = pooled_embed
        prompt_emb['text_ids'] = torch.zeros(visual_emb.shape[0], visual_emb.shape[1], 3).to(device=self.device, dtype=visual_emb.dtype)

        if "latents" in batch:
            latents = batch["latents"].to(dtype=self.pipe.torch_dtype, device=self.device)
        else:
            latents = self.pipe.vae_encoder(image.to(dtype=self.pipe.torch_dtype, device=self.device))

        ref_latents = self.pipe.vae_encoder(ref_image.to(dtype=self.pipe.torch_dtype, device=self.device))

        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(self.device)
        extra_input = self.pipe.prepare_extra_input(latents, guidance=3.5)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        all_latents = torch.concat([noisy_latents, ref_latents], dim=-2)
        ref_image_ids = self.pipe.denoising_model().prepare_image_ids(ref_latents)
        extra_input['image_ids'] = torch.concat([extra_input['image_ids'], ref_image_ids], dim=-2)

        # Compute loss
        noise_pred = self.pipe.denoising_model()(
            all_latents, timestep=timestep, **prompt_emb, **extra_input,
            use_gradient_checkpointing=self.use_gradient_checkpointing
        )
        noise_pred = noise_pred[:, :, :noisy_latents.shape[2], :]
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        # Record log
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        # Record log
        self.log("train_learning_rate", lr, prog_bar=True, on_step=True, on_epoch=False, rank_zero_only=True)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False, rank_zero_only=True)
        return loss


    def configure_optimizers(self):
        import itertools
        trainable_modules = itertools.chain(
            self.adapter.parameters(),
            self.global_adapter.parameters(),
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

        checkpoint.update(self.pipe.denoising_model().state_dict())

        global_adapter_state_dict = self.global_adapter.state_dict()
        global_prefix = 'global_adapter'
        global_adapter_state_dict = {f"{global_prefix}.{key}": value for key, value in global_adapter_state_dict.items()}
        checkpoint.update(global_adapter_state_dict)

        adapter_state_dict = self.adapter.state_dict()
        adapter_prefix = 'adapter'
        adapter_state_dict = {f"{adapter_prefix}.{key}": value for key, value in adapter_state_dict.items()}
        checkpoint.update(adapter_state_dict)



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
        num_workers=args.dataloader_num_workers
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
