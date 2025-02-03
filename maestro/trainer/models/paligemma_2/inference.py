import torch
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

from maestro.trainer.common.utils.device import parse_device_spec


def predict_with_inputs(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    device: torch.device,
    max_new_tokens: int = 1024,
) -> list[str]:
    """Generate text predictions from preprocessed model inputs.

    Args:
        model (PaliGemmaForConditionalGeneration): The PaliGemma model for generation.
        processor (PaliGemmaProcessor): Tokenizer and processor for model inputs/outputs.
        input_ids (torch.Tensor): Input token IDs.
        attention_mask (torch.Tensor): Attention mask for input tokens.
        pixel_values (torch.Tensor): Processed image tensor.
        device (torch.device): Device to run inference on.
        max_new_tokens (int): Maximum number of new tokens to generate.

    Returns:
        list[str]: List of generated text predictions.
    """
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=pixel_values.to(device),
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_new_tokens=max_new_tokens,
        )
        prefix_length = input_ids.shape[-1]
        generated_ids = generated_ids[:, prefix_length:]
        return processor.batch_decode(generated_ids, skip_special_tokens=True)


def predict(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    image: str | bytes | Image.Image,
    prefix: str,
    device: str | torch.device = "auto",
    max_new_tokens: int = 1024,
) -> str:
    """Generate a text prediction for a single image and text prefix.

    Args:
        model (PaliGemmaForConditionalGeneration): The PaliGemma model for generation.
        processor (PaliGemmaProcessor): Tokenizer and processor for model inputs/outputs.
        image (str | bytes | Image.Image): Input image as a file path, bytes, or PIL Image.
        prefix (str): Text prefix to condition the generation.
        device (str | torch.device): Device to run inference on.
        max_new_tokens (int): Maximum number of new tokens to generate.

    Returns:
        str: Generated text prediction.
    """
    device = parse_device_spec(device)
    text = "<image>" + prefix
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
    return predict_with_inputs(
        **inputs, model=model, processor=processor, device=device, max_new_tokens=max_new_tokens
    )[0]
