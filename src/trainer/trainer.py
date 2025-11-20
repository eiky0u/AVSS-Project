import torch

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            if not hasattr(self, "_accum_step"):
                self._accum_step = 0
            if self._accum_step == 0:
                self.optimizer.zero_grad(set_to_none=True)

        device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"

        with torch.autocast(
            device_type=device_type, dtype=self.amp_dtype, enabled=self.amp
        ):
            outputs = self.model(**batch)
            batch.update(outputs)
            all_losses = self.criterion(**batch)
            batch.update(all_losses)

        if self.is_train:
            do_update = (self._accum_step + 1) % self.accum == 0
            did_update = False

            if self.amp and self.scaler is not None:
                self.scaler.scale(batch["loss"] / self.accum).backward()
                if do_update:
                    self.scaler.unscale_(self.optimizer)
                    pre, post = self._clip_grad_norm()
                    self._last_logged_clipped_norm = float(post)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self._accum_step = 0
                    did_update = True
                else:
                    self._accum_step += 1
            else:
                (batch["loss"] / self.accum).backward()
                if do_update:
                    pre, post = self._clip_grad_norm()
                    self._last_logged_clipped_norm = float(post)

                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self._accum_step = 0
                    did_update = True
                else:
                    self._accum_step += 1

            self._last_do_update = did_update

        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        if mode == "train":
            self.log_audio(**batch)
            self.log_mouth(**batch)
        else:
            self.log_audio(**batch)
            self.log_mouth(**batch)
            self.log_prediction(**batch)

    def log_audio(
        self,
        mix,
        mix_len,
        targets,
        **batch,
    ):
        sr = 16000
        T = int(mix_len[0])
        self.writer.add_audio("mix", mix[0, :, :T].float(), sr)
        self.writer.add_audio("s1", targets[0, 0, :T].float(), sr)
        self.writer.add_audio("s2", targets[0, 1, :T].float(), sr)

    def log_mouth(
        self,
        mouths,
        mouths_len,
        **batch,
    ):
        mouths = mouths * 255.0
        FPS = 25
        F = int(mouths_len[0])
        self.writer.add_video("mouth1", mouths[0, 0, :F, :, :], fps=FPS)
        self.writer.add_video("mouth2", mouths[0, 1, :F, :, :], fps=FPS)

    def log_prediction(
        self,
        mix,
        mix_len,
        mouths,
        **batch,
    ):
        sr = 16000
        T = int(mix_len[0])
        device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        with torch.autocast(
            device_type=device_type, dtype=self.amp_dtype, enabled=self.amp
        ):
            preds = self.model(mix=mix, mouths=mouths)["preds"]
        self.writer.add_audio("pred_s1", preds[0, 0, :T].float(), sr)
        self.writer.add_audio("pred_s2", preds[0, 1, :T].float(), sr)
