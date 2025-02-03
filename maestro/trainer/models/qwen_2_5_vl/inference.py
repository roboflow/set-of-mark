import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

from maestro.trainer.common.utils.device import parse_device_spec
from maestro.trainer.models.qwen_2_5_vl.loaders import format_conversation


def predict_with_inputs(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    device: torch.device,
    max_new_tokens: int = 1024,
) -> list[str]:
    """
    Generates predictions from the Qwen2.5-VL model using both textual and visual inputs.

    Args:
        model (Qwen2_5_VLForConditionalGeneration):
            A Qwen2.5-VL model capable of conditional text generation with visual context.
        processor (Qwen2_5_VLProcessor):
            Preprocessing and postprocessing utility for the Qwen2.5-VL model.
        input_ids (torch.Tensor):
            Tokenized input text IDs.
        attention_mask (torch.Tensor):
            Attention mask corresponding to the tokenized input.
        pixel_values (torch.Tensor):
            Preprocessed image data (pixel values) for visual inputs.
        image_grid_thw (torch.Tensor):
            Tensor specifying the layout or shape of the provided images.
        device (torch.device):
            Device on which to run inference (e.g., ``torch.device("cuda")`` or ``torch.device("cpu")``).
        max_new_tokens (int):
            Maximum number of tokens to generate.

    Returns:
        list[str]: A list of decoded strings corresponding to the generated sequences.
    """
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            pixel_values=pixel_values.to(device),
            image_grid_thw=image_grid_thw.to(device),
            max_new_tokens=max_new_tokens,
        )
        generated_ids = [
            generated_sequence[len(input_sequence) :]
            for input_sequence, generated_sequence in zip(input_ids, generated_ids)
        ]
        return processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def predict(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
    image: str | bytes | Image.Image,
    prefix: str,
    system_message: str | None = None,
    device: str | torch.device = "auto",
    max_new_tokens: int = 1024,
) -> str:
    """
    Generates a single prediction from the Qwen2.5-VL model given an image and prefix text.

    Args:
        model (Qwen2_5_VLForConditionalGeneration):
            A Qwen2.5-VL model capable of conditional text generation with visual context.
        processor (Qwen2_5_VLProcessor):
            Preprocessing and postprocessing utility for the Qwen2.5-VL model.
        image (str | bytes | PIL.Image.Image):
            Image input for the model, which can be a file path, raw bytes, or a PIL Image object.
        prefix (str):
            Text prompt or initial text to prepend to the conversation.
        system_message (str | None):
            A system-level instruction or context text.
        device (str | torch.device):
            Device on which to run inference. Can be ``torch.device`` or a string such
            as "auto", "cpu", "cuda", or "mps".
        max_new_tokens (int):
            Maximum number of tokens to generate.

    Returns:
        str: The decoded string representing the model's generated response.
    """
    device = parse_device_spec(device)
    conversation = format_conversation(image=image, prefix=prefix, system_message=system_message)
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(conversation)

    inputs = processor(
        text=text,
        images=image_inputs,
        return_tensors="pt",
    )
    return predict_with_inputs(
        **inputs, model=model, processor=processor, device=device, max_new_tokens=max_new_tokens
    )[0]
