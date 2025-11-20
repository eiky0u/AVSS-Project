import math
import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="train_16")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    logger.info(model)

    # get loss and metrics
    loss_function = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)

    accum = getattr(config.trainer, "grad_accum_steps", 1)
    steps_per_epoch = math.ceil(len(dataloaders["train"]) / accum)
    lr_scheduler = instantiate(
        config.lr_scheduler,
        optimizer=optimizer,
        steps_per_epoch=steps_per_epoch,
    )

    # AMP
    use_amp = bool(config.trainer.get("use_amp", False)) and str(device).startswith(
        "cuda"
    )
    amp_dtype_name = str(config.trainer.get("amp_dtype", "fp16")).lower()
    amp_dtype = torch.float16 if amp_dtype_name == "fp16" else torch.bfloat16
    if use_amp and amp_dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("bf16 is not supported, switched to fp16")
        amp_dtype = torch.float16
    scaler = GradScaler(enabled=use_amp and amp_dtype is torch.float16)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
        amp=use_amp,
        scaler=scaler,
        amp_dtype=amp_dtype,
        accum=accum,
    )

    trainer.train()


if __name__ == "__main__":
    main()
