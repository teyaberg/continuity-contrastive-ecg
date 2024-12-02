import os

import hydra
import rootutils
from hydra.utils import instantiate
from omegaconf import DictConfig

from continuity_contrastive_ecg.utils.instantiators import (
    instantiate_callbacks,
    instantiate_loggers,
)

os.environ["PROJECT_ROOT"] = str(rootutils.setup_root(search_from=__file__, indicator=".project-root"))


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    datasets = instantiate(cfg.data)
    model = instantiate(cfg.model)
    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    lightning_logger = instantiate_loggers(cfg.get("logger"))
    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=lightning_logger)
    trainer.fit(model, datasets.train_loader, datasets.val_loader)
    trainer.test(model, datasets.test_loader)

    return model.min_val_loss


if __name__ == "__main__":
    main()
