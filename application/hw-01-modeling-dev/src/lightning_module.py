import pytorch_lightning as pl
import torch
from timm import create_model

from src.config import Config
from src.losses import get_losses
from src.metrics import get_metrics
from src.train_utils import load_object


class PlanetsModule(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self._config = config

        self._model = create_model(
            num_classes=self._config.num_classes,
            **self._config.train_config.model_kwargs,
        )
        self._losses = get_losses(self._config.losses)
        metrics = get_metrics(
            num_classes=self._config.num_classes,
            num_labels=self._config.num_classes,
            task="multilabel",
            average="macro",
            threshold=0.5,
        )
        self._valid_metrics = metrics.clone(prefix="val_")
        self._test_metrics = metrics.clone(prefix="test_")

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    def configure_optimizers(self):
        optimizer = load_object(self._config.train_config.optimizer)(
            self._model.parameters(),
            **self._config.train_config.optimizer_kwargs,
        )
        scheduler = load_object(
            self._config.train_config.scheduler,
        )(optimizer, **self._config.train_config.scheduler_kwargs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self._config.monitor_metric,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def training_step(self, batch, batch_idx):
        """
        Считаем лосс.
        """
        images, gt_labels = batch
        pr_logits = self(images)
        return self._calculate_loss(pr_logits, gt_labels, "train_")

    def validation_step(self, batch, batch_idx):
        """
        Считаем лосс и метрики.
        """
        images, gt_labels = batch
        pr_logits = self(images)
        pr_labels = torch.sigmoid(pr_logits)
        self._valid_metrics(pr_labels, gt_labels)

    def test_step(self, batch, batch_idx):
        """
        Считаем метрики.
        """
        images, gt_labels = batch
        pr_logits = self(images)
        pr_labels = torch.sigmoid(pr_logits)
        self._test_metrics(pr_labels, gt_labels)

    def on_validation_epoch_start(self) -> None:
        self._valid_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self._valid_metrics.compute(), on_epoch=True)

    def on_test_epoch_end(self) -> None:
        self.log_dict(self._test_metrics.compute(), on_epoch=True)

    def _calculate_loss(
        self,
        pr_logits: torch.Tensor,
        gt_labels: torch.Tensor,
        prefix: str,
    ) -> torch.Tensor:
        total_loss = 0
        for cur_loss in self._losses:
            loss = cur_loss.loss(pr_logits, gt_labels)
            total_loss += cur_loss.weight * loss
            self.log(f"{prefix}{cur_loss.name}_loss", loss.item())
        self.log(f"{prefix}total_loss", total_loss.item())
        return total_loss
