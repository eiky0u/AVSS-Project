import torch
from torch import nn
from tqdm.auto import tqdm
import torchaudio

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer

from thop import profile as thop_profile
import time
from pathlib import Path


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        model_type,
        ve,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.model_type = model_type
        self.ve = ve
        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

        self.sample_rate = 16000

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        if self.model_type == "tdfnet":
            outputs = self.model(**batch)

        elif self.model_type == "rtfsnet":
            lengths = [batch["mouths"][:, 0, :].size(2)] * batch["mouths"][
                :, 0, :
            ].size(0)
            v0_0 = self.ve(
                batch["mouths"][:, 0, :].unsqueeze(1).to(self.device), lengths=lengths
            )
            v0_1 = self.ve(
                batch["mouths"][:, 1, :].unsqueeze(1).to(self.device), lengths=lengths
            )
            s1 = self.model(v0_0, batch["mix"][:, 0, :]).to(self.device)
            s2 = self.model(v0_1, batch["mix"][:, 0, :]).to(self.device)
            outputs = {"preds": torch.stack([s1, s2], dim=1)}

        batch.update(outputs)

        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        preds = batch["preds"].clone().detach().cpu()  # [B, S, T]
        max_val_s1 = preds[:, 0, :].abs().max()
        max_val_s2 = preds[:, 1, :].abs().max()
        if max_val_s1 > 1.0:
            preds[:, 0, :] = preds[:, 0, :] / max_val_s1
        if max_val_s2 > 1.0:
            preds[:, 1, :] = preds[:, 1, :] / max_val_s2

        batch_size = preds.shape[0]
        for i in range(batch_size):
            utt = batch["utt"][i]
            s1 = preds[i, 0, :]  # [S, T]
            s2 = preds[i, 1, :]  # [S, T]

            if self.save_path is not None:
                torchaudio.save(
                    str(self.save_path / part / "s1" / f"{utt}.wav"),
                    s1.unsqueeze(0),
                    sample_rate=self.sample_rate,
                )
                torchaudio.save(
                    str(self.save_path / part / "s2" / f"{utt}.wav"),
                    s2.unsqueeze(0),
                    sample_rate=self.sample_rate,
                )

        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        if self.evaluation_metrics is not None:
            self.evaluation_metrics.reset()

        # create Save dir
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)
            (self.save_path / part / "s1").mkdir(exist_ok=True, parents=True)
            (self.save_path / part / "s2").mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        if self.evaluation_metrics is not None:
            return self.evaluation_metrics.result()
        else:
            return {}

    def run_speed_benchmark(self):
        """
        Run a benchmark on a single batch from the first dataloader.

        - Single forward pass (for timing & memory)
        - No saving to disk
        - No metrics
        - Reports:
            * time for one step
            * peak memory for that batch
            * MACs (if thop is installed)
            * total & trainable params (model [+ ve])
            * checkpoint size on disk
        """
        self.is_train = False
        self.model.eval()

        # take first dataloader
        _, dataloader = next(iter(self.evaluation_dataloaders.items()))
        try:
            batch = next(iter(dataloader))
        except StopIteration:
            print("Empty dataloader, cannot run speed benchmark.")
            return {}

        # move to device & apply transforms
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        # batch size
        batch_size = None
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                batch_size = v.size(0)
                break

        # parameter counts (model + optional video encoder)
        modules = [self.model]
        if self.ve is not None:
            modules.append(self.ve)

        total_params = sum(p.numel() for m in modules for p in m.parameters())
        trainable_params = sum(
            p.numel() for m in modules for p in m.parameters() if p.requires_grad
        )

        # checkpoint size
        ckpt = self.config.inferencer.get("from_pretrained", None)
        model_size_bytes = None
        model_size_mb = None
        if ckpt is not None:
            path = Path(ckpt)
            if path.is_file():
                model_size_bytes = path.stat().st_size
                model_size_mb = model_size_bytes / (1024**2)

        macs = None

        if self.model_type == "tdfnet":

            class TDFWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, batch_dict):
                    return self.model(**batch_dict)

            wrapper = TDFWrapper(self.model).to(self.device)
            macs, _ = thop_profile(wrapper, inputs=(batch,), verbose=False)

        elif self.model_type == "rtfsnet":

            class RTFSWrapper(nn.Module):
                def __init__(self, model, ve, device):
                    super().__init__()
                    self.model = model
                    self.ve = ve
                    self.device = device

                def forward(self, batch_dict):
                    mouths = batch_dict["mouths"]
                    mix = batch_dict["mix"]
                    lengths = [mouths[:, 0, :].size(2)] * mouths[:, 0, :].size(0)
                    v0_0 = self.ve(
                        mouths[:, 0, :].unsqueeze(1).to(self.device),
                        lengths=lengths,
                    )
                    v0_1 = self.ve(
                        mouths[:, 1, :].unsqueeze(1).to(self.device),
                        lengths=lengths,
                    )
                    s1 = self.model(v0_0, mix[:, 0, :])
                    s2 = self.model(v0_1, mix[:, 0, :])
                    return torch.stack([s1, s2], dim=1)

            wrapper = RTFSWrapper(self.model, self.ve, self.device).to(self.device)
            macs, _ = thop_profile(wrapper, inputs=(batch,), verbose=False)

        # timing & memory
        device = self.device
        use_cuda = torch.cuda.is_available() and "cuda" in str(device)

        if use_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device=device)
            torch.cuda.synchronize(device)

        with torch.no_grad():
            start = time.perf_counter()

            if self.model_type == "tdfnet":
                _ = self.model(**batch)
            elif self.model_type == "rtfsnet":
                lengths = [batch["mouths"][:, 0, :].size(2)] * batch["mouths"][
                    :, 0, :
                ].size(0)
                v0_0 = self.ve(
                    batch["mouths"][:, 0, :].unsqueeze(1).to(self.device),
                    lengths=lengths,
                )
                v0_1 = self.ve(
                    batch["mouths"][:, 1, :].unsqueeze(1).to(self.device),
                    lengths=lengths,
                )
                _ = self.model(v0_0, batch["mix"][:, 0, :])
                _ = self.model(v0_1, batch["mix"][:, 0, :])
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")

            if use_cuda:
                torch.cuda.synchronize(device)
            end = time.perf_counter()

        step_time = end - start
        throughput_examples = (
            batch_size / step_time if batch_size is not None and step_time > 0 else None
        )
        max_memory_bytes = (
            torch.cuda.max_memory_allocated(device=device) if use_cuda else None
        )
        max_memory_mb = (
            max_memory_bytes / (1024**2) if max_memory_bytes is not None else None
        )

        stats = {
            "batch_size": batch_size,
            "step_time_s": step_time,
            "throughput_examples_per_s": throughput_examples,
            "max_memory_bytes": max_memory_bytes,
            "max_memory_mb": max_memory_mb,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "model_size_bytes": model_size_bytes,
            "model_size_mb": model_size_mb,
            "macs": macs,
        }

        return stats
