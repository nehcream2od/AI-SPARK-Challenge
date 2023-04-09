import argparse
import collections
import os
import random

import data_module.data_module as module_data
import model.loss as module_loss
import model.model as module_arch
import model.optimizer as module_optimizer
import numpy as np
import pandas as pd
import torch
from data_module.data_module import CustomDataModule
from parse_config import ConfigParser
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from trainer.trainer import LitGANTrainer
from model.model import GeneratorWrapper

# Fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


def ensemble_models(fold_results):
    # This is just an example of averaging the fold results.
    # Modify this function according to the desired ensemble method.
    return np.mean(fold_results, axis=0)


def main(config):
    # Create data module instance
    data_module = config.init_obj("data_module", CustomDataModule)
    data_module.prepare_data()
    data_module.setup()

    ensemble_results = []

    for tp in range(len(data_module.train_datasets)):
        # print(f"Processing equipment type {tp + 1}/{len(data_module.train_datasets)}")
        tp_fold_results = []
        train_loaders, val_loaders = data_module.train_val_dataloader()

        for fold, (train_loader, val_loader) in enumerate(
            zip(train_loaders[tp], val_loaders[tp])
        ):
            # print(f"Fold {fold + 1}/{len(train_loaders[tp])}")
            input_size = train_loader.dataset[0].shape[0]
            # print(train_loader.dataset[0].shape)
            output_size = input_size

            config["arch"]["args"]["input_size"] = input_size
            config["arch"]["args"]["output_size"] = output_size

            # Build model architecture
            model = config.init_obj("arch", module_arch)

            # Compile model
            torch.compile(model)

            # Create a PyTorch Lightning trainer
            trainer = Trainer(
                accelerator=config["trainer"]["accelerator"],
                devices=config["trainer"]["devices"],
                max_epochs=config["trainer"]["max_epochs"],
                logger=WandbLogger(
                    project="gan",
                    name=config["name"],
                    config=config,
                    save_dir=config["trainer"]["save_dir"],
                ),
                log_every_n_steps=1,
            )
            lit_gan_trainer = LitGANTrainer(
                model=model,
                criterion_gen=getattr(module_loss, config["criterion_gen"]),
                criterion_disc=getattr(module_loss, config["criterion_disc"]),
                gen_optimizer_class=getattr(
                    module_optimizer
                    if config["gen_optimizer"]["type"] == "Lion"
                    else torch.optim,
                    config["gen_optimizer"]["type"],
                ),
                disc_optimizer_class=getattr(
                    module_optimizer
                    if config["disc_optimizer"]["type"] == "Lion"
                    else torch.optim,
                    config["disc_optimizer"]["type"],
                ),
                config=config,
                alpha=config["alpha"],
            )

            # Train on the current fold
            trainer.fit(
                lit_gan_trainer,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
            )

            # (Optional) Save fold-specific model checkpoint
            checkpoint_path = (
                f"{config['trainer']['save_dir']}/type_{tp + 1}_fold_{fold + 1}.ckpt"
            )
            trainer.save_checkpoint(checkpoint_path)

            # Evaluate the model on the prediction set
            wrapped_generator = GeneratorWrapper(model.generator)
            predict_loaders = data_module.predict_dataloader()
            tp_result = []

            for tp_loader in predict_loaders[tp]:
                result = trainer.predict(wrapped_generator, tp_loader)
                tp_result.append(result)
            tp_result = np.concatenate(tp_result, axis=0)
            tp_fold_results.append(tp_result)
        print(np.array(tp_fold_results).shape)

        # print(np.array(tp_fold_results).shape)
        # Ensemble fold results for the current equipment type
        # tp_ensemble_result = ensemble_models(tp_fold_results)
        # ensemble_results.append(tp_ensemble_result)

    # # Combine ensemble results for all equipment types
    # ensemble_results = np.concatenate(ensemble_results, axis=0)
    # print(pd.DataFrame(ensemble_results).shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    parser.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(
            ["--lr", "--learning_rate"],
            type=float,
            target=("gen_optimizer", "args", "lr"),
        ),
        CustomArgs(
            ["--bs", "--batch_size"],
            type=int,
            target=("data_module", "args", "batch_size"),
        ),
    ]
    config = ConfigParser.from_args(parser, options)
    main(config)
