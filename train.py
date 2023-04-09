import argparse
import collections
import os
import random
import data_module.data_module as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import numpy as np
import torch
from parse_config import ConfigParser
from trainer import LitGANTrainer
from utils import prepare_device
from pytorch_lightning import Trainer
from importlib import import_module
from data_module.data_module import CustomDataModule
from lightning.pytorch.loggers import WandbLogger
from trainer.trainer import LitGANTrainer


# fix random seeds for reproducibility
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
    data_module_args = config["data_module"]["args"]
    data_module = config.init_obj("data_module", CustomDataModule)

    data_module.prepare_data()
    data_module.setup()

    fold_results = []

    for fold, (train_loader, val_loader) in enumerate(
        data_module.train_val_dataloader()
    ):
        input_size = train_loader.dataset[0].shape[0]
        output_size = input_size

        # build model architecture
        model = config.init_obj(
            "arch",
            module_arch,
            input_size,
            output_size,
            # config["arch"]["args"]["gen_hidden_size"],
            # config["arch"]["args"]["dsc_hidden_size"],
        )

        # prepare for (multi-device) GPU training
        device, device_ids = prepare_device(config["n_gpu"])
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        # create a PyTorch Lightning trainer
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
        )
        lit_gan_trainer = LitGANTrainer(
            model=model,
            criterion_gen=getattr(module_loss, config["criterion_gen"]),
            criterion_disc=getattr(module_loss, config["criterion_dsc"]),
            config=config,
            alpha=0.5,
        )

        # Train on the current fold
        trainer.fit(
            lit_gan_trainer, train_dataloaders=train_loader, val_dataloaders=val_loader
        )

        # (Optional) Save fold-specific model checkpoint
        checkpoint_path = f"{config['save_dir']}/fold_{fold}.ckpt"
        trainer.save_checkpoint(checkpoint_path)

        # Evaluate the model on the validation set
        result = trainer.validate(model, val_dataloaders=val_loader)
        fold_results.append(result)

    # Ensemble the models using the fold_results
    ensemble_result = ensemble_models(fold_results)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
