from typing import Literal, Union

from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, PaliGemmaForConditionalGeneration

LoraInitLiteral = Literal["gaussian", "olora", "pissa", "pissa_niter_[number of iters]", "loftq"]
# TO DO
# Make revision as RevisionLiteral?


def prepare_peft_model(
    model: Union[AutoModelForCausalLM, PaliGemmaForConditionalGeneration],
    r: int = 8,
    lora_alpha: int = 8,
    lora_dropout: float = 0.05,
    bias: Literal["none", "all", "lora_only"] = "none",
    inference_mode: bool = False,
    use_rslora: bool = True,
    init_lora_weights: Union[bool, LoraInitLiteral] = "gaussian",
    revision: str = "",
) -> PeftModel:
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "linear", "Conv2d", "lm_head", "fc2"],
        task_type="CAUSAL_LM",
        lora_dropout=lora_dropout,
        bias=bias,
        inference_mode=inference_mode,
        use_rslora=use_rslora,
        init_lora_weights=init_lora_weights,
        revision=revision,
    )
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    return peft_model.to(model.device)
