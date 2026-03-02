from pathlib import Path
import logging
import os
import copy
import math
import shutil
from tqdm.auto import tqdm
from contextlib import nullcontext
import wandb

from PIL import Image
import numpy as np
import torch
import torchvision
import diffusers
import transformers
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
from parser_helper import parse_args
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    Flux2KleinPipeline,
    Flux2Transformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _collate_lora_metadata,
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    find_nearest_bucket,
    free_memory,
    parse_buckets_string,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module
from data_module import KleinDataset, collate_fn
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from utils import encode_prompt,prepare_latents, prepare_image_latents_batch

check_min_version("0.37.0.dev0")
logger = get_logger(__name__)

args = parse_args()


def log_validation(
    pipeline,
    args,
    accelerator,
    dataloader,
    tag,
    is_final_validation=False,
    prompt_embeds=None,
):
    logger.info(f"Running {tag}... \n ")
    pipeline = pipeline.to(accelerator.device)
    
    # Use appropriate precision context
    if accelerator.mixed_precision == "bf16":
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    elif accelerator.mixed_precision == "fp16":
        autocast_ctx = torch.autocast("cuda", dtype=torch.float16)
    else:
        autocast_ctx = nullcontext()

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    
    with autocast_ctx:
        images = []
        prompts = []
        control_images = []
        target_images = []
        
        # Process full dataset - no limit on samples for validation
        # This ensures we use all images from the test dataset
        
        for batch in dataloader:
            prompt = batch['prompts']
            control_image = batch['source_image']
            target_image = batch['target_image']
            
            # for klein there may be a bug for PIL input only, so 1 validation_bsz for now - to be fixed in the future
            validation_batch_size = 1
            
            for i in range(0, len(prompt), validation_batch_size):
                batch_prompt = prompt[i:i+validation_batch_size]
                batch_control_image = control_image[i:i+validation_batch_size]
                batch_target_image = target_image[i:i+validation_batch_size]
            
                control_image = batch_control_image * 0.5 + 0.5
                control_image = control_image.clamp(0, 1).squeeze(0)# only for validation_bsz=1 - to be fixed in the future
                #to PIL
                input_control_images = []
                for j in range(control_image.shape[0]):
                    input_control_images.append(torchvision.transforms.functional.to_pil_image(control_image[j].cpu()))
                if not args.same_prompt_for_all:
                    result = pipeline(
                        prompt=batch_prompt,
                        height=args.height,
                        width=args.width,
                        image=input_control_images, 
                        num_inference_steps=10,
                        generator=generator,
                        guidance_scale=1,
                    ).images
                else:
                    result = pipeline(
                        prompt=None,
                        height=args.height,
                        width=args.width,
                        image=input_control_images,
                        num_inference_steps=10,
                        generator=generator,
                        guidance_scale=1,
                        prompt_embeds=prompt_embeds,
                    ).images 
                
                images.extend(result)
                prompts.extend(batch_prompt)
                control_images.extend(batch_control_image)
                target_images.extend(batch_target_image)

    # Log to trackers
    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            formatted_images = []
            
            for input_img, gen_img, tgt_img, prompt in zip(control_images, images, target_images, prompts):
                input_img = input_img.permute(1, 2, 0, 3).contiguous().view(input_img.shape[1], input_img.shape[2], input_img.shape[0] * input_img.shape[3])
                input_img_pil = Image.fromarray((((input_img.cpu().numpy() + 1) / 2).transpose(1, 2, 0) * 255).astype(np.uint8))
                tgt_img_pil = Image.fromarray((((tgt_img.cpu().numpy()+1)/2).transpose(1, 2, 0) * 255).astype(np.uint8))
                gen_img_pil = gen_img.convert('RGB')
                
                total_width = input_img_pil.width + tgt_img_pil.width + gen_img_pil.width
                max_height = max(input_img_pil.height, tgt_img_pil.height, gen_img_pil.height)
                
                combined_img = Image.new('RGB', (total_width, max_height))
                
                combined_img.paste(input_img_pil, (0, 0))
                combined_img.paste(tgt_img_pil, (input_img_pil.width, 0))
                combined_img.paste(gen_img_pil, (input_img_pil.width + tgt_img_pil.width, 0))
                
                formatted_images.append(wandb.Image(combined_img, caption=f"Source | Target | Generated: {prompt}"))
            
            tracker.log({tracker_key: formatted_images})
    del images, prompts, control_images, target_images
    del pipeline
    free_memory()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = Qwen2TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    vae = AutoencoderKLFlux2.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = Flux2Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )
    num_channels_latents = transformer.config.in_channels // 4
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # use regex because to_out are modulelist in dual-stream but linear in single-stream
    target_modules = r"""
        .*\.attn\.to_[qkv]|
        single_transformer_blocks\..*\.attn\.to_out|
        transformer_blocks\..*\.attn\.to_out\.0|
        .*\.attn\.add_[qkv]_proj|
        .*\.attn\.to_add_out|
        .*\.attn\.to_qkv_mlp_proj|
        .*\.ff(?:_context)?\.linear_(in|out)
    """.replace("\n", "").replace(" ", "")


    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            modules_to_save = {}
            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    modules_to_save["transformer"] = model
                elif isinstance(model, type(unwrap_model(text_encoder))):
                    text_encoder_one_lora_layers_to_save = get_peft_model_state_dict(model)
                    modules_to_save["text_encoder"] = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            Flux2KleinPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                **_collate_lora_metadata(modules_to_save),
            )

    def load_model_hook(models, input_dir):
        transformer_ = None
        text_encoder_one_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            elif isinstance(model, type(unwrap_model(text_encoder))):
                text_encoder_one_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = Flux2KleinPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )
        if args.train_text_encoder and args.text_encoder_lr:
            logger.warning(
                f"Learning rates were provided both for the transformer and the text encoder- e.g. text_encoder_lr:"
                f" {args.text_encoder_lr} and learning_rate: {args.learning_rate}. "
                f"When using prodigy only learning_rate is used as the initial learning rate."
            )
            # changes the learning rate of text_encoder_parameters_one to be
            # --learning_rate
            params_to_optimize[1]["lr"] = args.learning_rate

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    train_dataset = KleinDataset(
        size=(args.width, args.height),
        split="train",  # Training split
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        batch_size=args.train_batch_size,
        shuffle=True,
    )

    if args.validation_check:
        validation_dataset = KleinDataset(
            size=(args.width, args.height),
            split="test",  # Test split
        )

        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            collate_fn=collate_fn,
            num_workers=args.dataloader_num_workers,
        )


    def compute_text_embeddings(prompt, text_encoder, tokenizer):
        with torch.no_grad():
            prompt_embeds, text_ids, raw_prompt_embeds = encode_prompt(
                text_encoder, tokenizer, prompt
            )
            prompt_embeds = prompt_embeds.to(accelerator.device)
            text_ids = text_ids.to(accelerator.device)
        return prompt_embeds, text_ids, raw_prompt_embeds

    if args.same_prompt_for_all:
        instance_prompt_hidden_states, instance_text_ids, instance_raw_prompt_embeds = compute_text_embeddings(
            args.single_prompt, text_encoder, tokenizer
        )
        del text_encoder, tokenizer
        torch.cuda.empty_cache()

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * accelerator.num_processes * num_update_steps_per_epoch
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "Klein-FineTuning"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate):
                prompts = batch["prompts"]
                batch_size = batch["target_image"].shape[0]
                # encode batch prompts when custom prompts are provided for each image -
                if not args.same_prompt_for_all:
                    prompt_embeds, text_ids, _ = encode_prompt(
                        text_encoder, tokenizer, prompts, accelerator.device
                    )
                else:
                    elems_to_repeat = len(prompts)
                    prompt_embeds = instance_prompt_hidden_states.repeat(elems_to_repeat, 1, 1)
                    text_ids = instance_text_ids.repeat(elems_to_repeat, 1, 1)
                
                # Convert images to latent space
                pixel_values = batch["target_image"].to(dtype=vae.dtype) # target image
                source_pixel_values = batch["source_image"].to(dtype=vae.dtype) # source image
                vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
                noise, latent_ids = prepare_latents(
                    batch_size=batch_size,
                    num_latents_channels=num_channels_latents,
                    height=args.height,
                    width=args.width,
                    dtype=prompt_embeds.dtype,
                    device=accelerator.device,
                    vae_scale_factor=vae_scale_factor,
                    latents=None,
                )
                sources_latents, sources_latent_ids = prepare_image_latents_batch(
                    vae=vae,
                    images=source_pixel_values,
                    batch_size=batch_size,
                    device=accelerator.device,
                    dtype=vae.dtype,
                )
                gt_latents, _ = prepare_image_latents_batch(
                    vae=vae,
                    images=pixel_values.unsqueeze(1), # pretend to ref=1, B x refs x C x H x W
                    batch_size=batch_size,
                    device=accelerator.device,
                    dtype=vae.dtype,
                )

                latent_image_ids = torch.cat([latent_ids, sources_latent_ids], dim=1) 

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=batch_size,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=gt_latents.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=gt_latents.ndim, dtype=gt_latents.dtype)
                noisy_model_input = (1.0 - sigmas) * gt_latents + sigmas * noise #adding noise on the target image
                # print(f"noisy_model_input shape: {noisy_model_input.shape}")

                # handle guidance
                if unwrap_model(transformer).config.guidance_embeds:
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(gt_latents.shape[0])
                else:
                    guidance = None

                # concatenate the packed noisy model input with the packed source model input across channels dimensions
                transformer_input = torch.cat(
                    [noisy_model_input, sources_latents], dim=1
                )

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=transformer_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]

                # noise on only the target image
                model_pred = model_pred[:, : noisy_model_input.size(1)]

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                target = noise - gt_latents

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        transformer.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
            
                # Run validation every args.validation_steps steps (e.g., every 5 steps)
                if global_step % args.validation_steps == 0:
                    if args.same_prompt_for_all:
                        pipeline = Flux2KleinPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            transformer=accelerator.unwrap_model(transformer),
                            torch_dtype=weight_dtype,
                            vae=vae,
                            tokenizer=None,
                            text_encoder=None
                        )
                        log_validation(
                            pipeline=pipeline,
                            args=args,
                            accelerator=accelerator,
                            dataloader=validation_dataloader,
                            tag="validation",
                            prompt_embeds=instance_raw_prompt_embeds,
                        )
                    else:
                        pipeline = Flux2KleinPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            transformer=accelerator.unwrap_model(transformer),
                            torch_dtype=weight_dtype,
                            vae=vae,
                            tokenizer=tokenizer,
                            text_encoder=text_encoder
                        )
                        log_validation(
                            pipeline=pipeline,
                            args=args,
                            accelerator=accelerator,
                            dataloader=validation_dataloader,
                            tag="validation",
                        )

            if global_step >= args.max_train_steps:
                break

if __name__ == "__main__":
    args = parse_args()
    main(args)
