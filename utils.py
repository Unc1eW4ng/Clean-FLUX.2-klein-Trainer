from diffusers import Flux2KleinPipeline
import torch


def encode_prompt(
    text_encoder,
    tokenizer,
    prompt: str | list[str],
    device: torch.device | None = None,
    num_images_per_prompt: int = 1,
    prompt_embeds: torch.Tensor | None = None,
    max_sequence_length: int = 512,
    text_encoder_out_layers: tuple[int] = (9, 18, 27),
):
    device=device if device is not None else text_encoder.device

    if prompt is None:
        prompt = ""

    prompt = [prompt] if isinstance(prompt, str) else prompt

    if prompt_embeds is None:
        prompt_embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_sequence_length=max_sequence_length,
            hidden_states_layers=text_encoder_out_layers,
        )
        raw_prompt_embeds = prompt_embeds

    batch_size, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    text_ids = Flux2KleinPipeline._prepare_text_ids(prompt_embeds)
    text_ids = text_ids.to(device)
    return prompt_embeds, text_ids, raw_prompt_embeds


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: torch.Generator | None = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._encode_vae_image
def _encode_vae_image(vae,image: torch.Tensor):
    if image.ndim != 4:
        raise ValueError(f"Expected image dims 4, got {image.ndim}.")

    image_latents = retrieve_latents(vae.encode(image), sample_mode="argmax")
    image_latents = Flux2KleinPipeline._patchify_latents(image_latents)

    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(image_latents.device, image_latents.dtype)
    latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
    image_latents = (image_latents - latents_bn_mean) / latents_bn_std

    return image_latents



def prepare_latents(
    batch_size,
    num_latents_channels,
    height,
    width,
    dtype,
    device,
    vae_scale_factor,
    latents: torch.Tensor | None = None,
):
    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    shape = (batch_size, num_latents_channels * 4, height // 2, width // 2)
    if latents is None:
        latents = torch.randn(shape, device=device, dtype=dtype)
    else:
        latents = latents.to(device=device, dtype=dtype)

    latent_ids = Flux2KleinPipeline._prepare_latent_ids(latents)
    latent_ids = latent_ids.to(device)

    latents = Flux2KleinPipeline._pack_latents(latents)  # [B, C, H, W] -> [B, H*W, C]
    return latents, latent_ids

def prepare_image_latents_batch(
    vae,
    images: torch.Tensor,# B x Refs x C x H x W
    batch_size,
    device,
    dtype,
):
    batch_size, num_refs, C, H, W = images.shape
    batch_latents = []
    batch_latents_ids = []
    for i in range(batch_size):
        sample_latent,sample_ids = prepare_image_latents(
            vae=vae,
            images=list(images[i]),  # (Refs, C, H, W) -> list of (C, H, W)
            device=device,
            dtype=dtype,
        )
        batch_latents.append(sample_latent)  # (1, Refs*1024, 128)
        batch_latents_ids.append(sample_ids)
    batch_latents = torch.cat(batch_latents, dim=0)  # (B, Refs*1024, 128)
    batch_latents_ids = torch.cat(batch_latents_ids, dim=0)  
    # breakpoint()
    return batch_latents, batch_latents_ids
    
# Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline.prepare_image_latents
def prepare_image_latents(
    vae,
    images: list[torch.Tensor],
    device,
    dtype,
):
    image_latents = []
    for image in images:
        image = image.to(device=device, dtype=dtype)
        if image.ndim != 4:
            image.unsqueeze_(0)  # add batch dim if missing
        imagge_latent = _encode_vae_image(vae,image=image)
        image_latents.append(imagge_latent)  # (1, 128, 32, 32)

    image_latent_ids = Flux2KleinPipeline._prepare_image_ids(image_latents)

    # Pack each latent and concatenate
    packed_latents = []
    for latent in image_latents:
        # latent: (1, 128, 32, 32)
        packed = Flux2KleinPipeline._pack_latents(latent)  # (1, 1024, 128)
        packed = packed.squeeze(0)  # (1024, 128) - remove batch dim
        packed_latents.append(packed)

    # Concatenate all reference tokens along sequence dimension
    image_latents = torch.cat(packed_latents, dim=0)  # (N*1024, 128)
    image_latents = image_latents.unsqueeze(0)  # (1, N*1024, 128)

    image_latent_ids = image_latent_ids.to(device)

    return image_latents, image_latent_ids
