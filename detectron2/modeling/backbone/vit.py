import torch
from peft import inject_adapter_in_model, LoraConfig
import logging
from .build import BACKBONE_REGISTRY
from lib.dinov2.vit import DinoVisionTransformer, vit_base, vit_large

logger = logging.getLogger("detectron2.backbone")

def find_all_linear_modules(model):
    '''
    COPIED FROM https://github. com/artidoro/glora/blob/main/qlora.py
    QLoRA paper recommends "LoRA on all linear transformer block layers is required to match full finetuning performance."
    '''
    lora_module_names = set ()
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear)):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

            if "lm_head" in lora_module_names: # needed for 16-bit
                lora_module_names.remove("lm_head")
    return list(lora_module_names)


@BACKBONE_REGISTRY.register()
def build_dino_v2_vit(cfg, input_shape):
    out_indices = cfg.DE.OUT_INDICES

    if out_indices is not None:
        if isinstance(out_indices, str):
            out_indices = [int(m) for m in out_indices.split(",")]
    
    if cfg.MODEL.BACKBONE.TYPE == 'small':
        model = DinoVisionTransformer(
        patch_size=14,
        img_size=518,
        init_values=1,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        out_indices=out_indices,
    )
    elif cfg.MODEL.BACKBONE.TYPE == 'base':
        model = vit_base(out_indices=out_indices)
    elif cfg.MODEL.BACKBONE.TYPE == "large":
        model = vit_large(img_size=518, patch_size=14, init_values=1, out_indices=out_indices)
    else:
        raise NotImplementedError()
    
    if cfg.MODEL.ADD_LORA is True:

        linear_layers = find_all_linear_modules(model)
        lora_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            target_modules=linear_layers,
        )

        model = inject_adapter_in_model(lora_config, model, adapter_name='lora')

        logger.info(f"Added LoRA adapters to {linear_layers}")
        print(f"Added LoRA adapters to {linear_layers}")


    return model