import json
import torch
from src.lipreading_pretrained.model import Lipreading
import gdown


def get_lipreading_model(
    config_path: str,
    num_classes: int = 500,
    modality: str = "video",
    pretrained: bool = True,
) -> torch.nn.Module:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    tcn_options = {}
    if cfg.get("tcn_kernel_size") or cfg.get("tcn_num_layers"):
        tcn_options = {
            "kernel_size": cfg.get("tcn_kernel_size", []),
            "num_layers": cfg.get("tcn_num_layers", cfg.get("tcn_num_layers", 0)),
            "dropout": cfg.get("tcn_dropout", 0.0),
            "dwpw": cfg.get("tcn_dwpw", False),
            "width_mult": cfg.get("tcn_width_mult", 1),
        }

    densetcn_options = {}
    if cfg.get("densetcn_block_config"):
        densetcn_options = {
            "block_config": cfg.get("densetcn_block_config"),
            "growth_rate_set": cfg.get("densetcn_growth_rate_set"),
            "reduced_size": cfg.get("densetcn_reduced_size"),
            "kernel_size_set": cfg.get("densetcn_kernel_size_set"),
            "dilation_size_set": cfg.get("densetcn_dilation_size_set"),
            "squeeze_excitation": cfg.get("densetcn_se", False),
            "dropout": cfg.get("densetcn_dropout", 0.0),
        }

    model = Lipreading(
        modality=modality,
        num_classes=num_classes,
        tcn_options=tcn_options if any(tcn_options.values()) else {},
        densetcn_options=densetcn_options if densetcn_options else {},
        backbone_type=cfg.get("backbone_type", "resnet"),
        relu_type=cfg.get("relu_type", "prelu"),
        width_mult=cfg.get("width_mult", 1.0),
        use_boundary=cfg.get("use_boundary", False),
        extract_feats=True,
    )
    if pretrained:
        url = "https://drive.google.com/uc?id=179NgMsHo9TeZCLLtNWFVgRehDvzteMZE"
        ckpt = torch.load(gdown.download(url, quiet=False), map_location="cpu")[
            "model_state_dict"
        ]
        model.load_state_dict(ckpt, strict=True)
    return model
