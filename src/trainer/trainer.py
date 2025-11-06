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
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
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
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
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
        self.writer.add_audio(f"mix", mix[0, :, :T], sr)
        self.writer.add_audio(f"s1", targets[0, 0, :T], sr)
        self.writer.add_audio(f"s2", targets[0, 1, :T], sr)

    def log_mouth(
        self,
        mouths,
        mouths_len,
        **batch,
    ):
        FPS = 25
        F = int(mouths_len[0])
        self.writer.add_video(f"mouth1", mouths[0, 0, :F, :, :], fps=FPS)
        self.writer.add_video(f"mouth2", mouths[0, 1, :F, :, :], fps=FPS)

    def log_prediction(
        self,
        mix,
        mix_len,
        **batch,
    ):
        sr = 16000
        T = int(mix_len[0])
        preds = self.model(mix)["preds"]
        self.writer.add_audio(f"pred_s1", preds[0, 0, :T], sr)
        self.writer.add_audio(f"pred_s2", preds[0, 1, :T], sr)
