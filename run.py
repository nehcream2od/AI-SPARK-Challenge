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
from model.model import GeneratorWrapper
from parse_config import ConfigParser
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from trainer.trainer import LitGANTrainer
from utils import flatten_batches
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


def main(config):
    # Create data module instance
    data_module = config.init_obj("data_module", CustomDataModule)
    data_module.prepare_data()
    data_module.setup()

    ensemble_results = []

    # prepare data loaders
    train_loaders, val_loaders = data_module.train_val_dataloader()
    predict_loaders = data_module.predict_dataloader()

    all_anomaly = []

    for tp in range(len(train_loaders)):
        tp_fold_results = []  # fold 별 예측 결과 -> 1개의 설비

        threshold_mean_per_fold = []
        mse_mean_per_fold = []

        # fold 시작 fold 별로 train, validation하며 모델 학습, 결과를 뽑아내야함
        for fold, (train_loader, val_loader) in enumerate(
            zip(train_loaders[tp], val_loaders[tp])
        ):
            input_size = train_loader.dataset[0].shape[0]
            output_size = input_size

            config["arch"]["args"]["input_size"] = input_size
            config["arch"]["args"]["output_size"] = output_size

            # Build model
            model = config.init_obj("arch", module_arch)

            # Compile
            torch.compile(model)

            # set checkpoint
            checkpoint_callback = ModelCheckpoint(
                monitor="val_generator_loss",
                mode="min",
                save_top_k=1,
                save_last=False,
                filename=f"{config['trainer']['save_dir']}/type_{tp + 1}_fold_{fold + 1}.ckpt",
                verbose=True,
            )

            # set early stopping
            early_stop_callback = EarlyStopping(
                monitor="val_generator_loss",
                min_delta=0.00,
                patience=100,
                verbose=True,
                mode="min",  # loss는 낮아야 좋음
            )

            # logger
            wandb_logger = WandbLogger(
                project="AI-SPARK-Challenge",
                name=config["name"],
                config=config,
                save_dir=config["trainer"]["save_dir"],
                log_model="all",
            )

            # Create PyTorch Lightning trainer
            trainer = Trainer(
                accelerator=config["trainer"]["accelerator"],
                devices=config["trainer"]["devices"],
                max_epochs=config["trainer"]["max_epochs"],
                logger=wandb_logger,
                log_every_n_steps=100,
                callbacks=[checkpoint_callback, early_stop_callback],
            )

            # Create Model trainer
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

            wrapped_generator = GeneratorWrapper(model.generator)

            # Calculate the reconstruction error for the current fold
            reconstructed = trainer.predict(wrapped_generator, train_loader)
            reconstructed = flatten_batches(reconstructed)
            train_data = flatten_batches(train_loader)
            mse = np.mean(np.square(train_data - reconstructed), axis=1)

            threshold_per_fold = np.percentile(
                mse, config["threshold"] * 100
            )  # 95% 이상이면 이상치라고 판단
            # threshold_per_fold = np.max(mse)
            threshold_mean_per_fold.append(threshold_per_fold)

            # predict
            reconstructed_test = trainer.predict(wrapped_generator, predict_loaders[tp])
            reconstructed_test = flatten_batches(reconstructed_test)
            test_data = flatten_batches(predict_loaders[tp])
            mse_test = np.mean(np.square(test_data - reconstructed_test), axis=1)
            mse_mean_per_fold.append(mse_test)

        threshold_mean = np.mean(threshold_mean_per_fold)
        mse_mean = np.mean(mse_mean_per_fold, axis=0)
        anomaly = mse_mean > threshold_mean
        all_anomaly.extend(anomaly)

    # submission
    sample = pd.read_csv("./data/answer_sample.csv")
    sample["label"] = all_anomaly
    error_len = sum(all_anomaly)
    sample.to_csv(
        f"./data/{config['name']}_err{error_len}.csv",
        index=False,
    )
    print(sample["label"].value_counts())


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
