import argparse
import collections
import os
import random
import sys

import data_module.data_module as module_data
import model.loss as module_loss
import model.model as module_arch
import model.optimizer as module_optimizer
import numpy as np
import pandas as pd
import torch
from data_module.data_module import CustomDataModule
from model.model import GeneratorWrapper, LitGANModel
from parse_config import ConfigParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from utils import flatten_batches

# Fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# 현재 폴더 모듈 경로 추가
sys.path.append(os.path.abspath("."))


def main(config):
    # Create data module instance
    data_module = config.init_obj("data_module", CustomDataModule)
    data_module.prepare_data()
    data_module.setup()

    # prepare data loaders
    train_loaders, val_loaders = data_module.train_val_dataloader()
    predict_loaders = data_module.predict_dataloader()

    ensemble_results = []

    for tp in range(len(train_loaders)):
        anomaly_per_fold = []

        # fold 시작 fold 별로 train, validation하며 모델 학습, 결과를 뽑아내야함
        for fold, (train_loader, val_loader) in enumerate(
            zip(train_loaders[tp], val_loaders[tp])
        ):
            # 변수 초기화
            if globals().get("model"):
                del model
            if globals().get("trainer"):
                del trainer
            if globals().get("lit_gan"):
                del lit_gan_trainer
            if globals().get("wrapped_generator"):
                del wrapped_generator

            # Create Model
            """config 통해서 모델 초기화 할 때 참고
            model = config.init_obj("arch", module_arch)"""

            lit_gan = LitGANModel(
                input_size=train_loader.dataset[0].shape[0],
                output_size=train_loader.dataset[0].shape[0],
                gen_hidden_size=config["arch"]["args"]["gen_hidden_size"],
                disc_hidden_size=config["arch"]["args"]["disc_hidden_size"],
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
                gen_lr=config["gen_optimizer"]["args"]["lr"],
                disc_lr=config["disc_optimizer"]["args"]["lr"],
                config=config,
                alpha=config["alpha"],
            )

            # compile
            torch.compile(lit_gan)

            # set checkpoint
            path = f"{config['trainer']['save_dir']}/type_{tp}_fold_{fold + 1}"
            checkpoint_callback = ModelCheckpoint(
                monitor="val_mse",
                mode="min",
                save_top_k=1,
                save_last=False,
                filename=path,
                verbose=True,
            )

            # set early stopping
            early_stop_callback = EarlyStopping(
                monitor="val_mse",
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
                log_model=False,
            )

            # Create PyTorch Lightning trainer
            trainer = Trainer(
                accelerator=config["trainer"]["accelerator"],
                devices=config["trainer"]["devices"],
                max_epochs=config["trainer"]["max_epochs"],
                logger=wandb_logger,
                log_every_n_steps=1000,
                callbacks=[checkpoint_callback, early_stop_callback, RichProgressBar()],
                detect_anomaly=True,
            )

            # Train on the current fold
            trainer.fit(
                lit_gan,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
            )

            # load best model
            wandb_save_dir = wandb_logger.save_dir
            run_id = wandb_logger.experiment.id
            lit_gan.load_state_dict(
                torch.load(
                    f"./{wandb_save_dir}AI-SPARK-Challenge/{run_id}/checkpoints/saved/type_{tp}_fold_{fold + 1}.ckpt"
                )["state_dict"]
            )

            wrapped_generator = GeneratorWrapper(lit_gan.generator)

            # Calculate the reconstruction error for the current fold
            with torch.no_grad():
                reconstructed = trainer.predict(
                    wrapped_generator,
                    train_loader,
                )
            reconstructed = flatten_batches(reconstructed)
            train_data = flatten_batches(train_loader)
            mse = np.mean(np.square(train_data - reconstructed), axis=1)

            threshold_per_fold = np.percentile(
                mse, config["threshold"] * 100
            )  # 이상치 판단 기준 * 100
            # threshold_per_fold = np.max(mse)

            # predict
            with torch.no_grad():
                reconstructed_test = trainer.predict(
                    wrapped_generator,
                    predict_loaders[tp],
                )
            reconstructed_test = flatten_batches(reconstructed_test)
            test_data = flatten_batches(predict_loaders[tp])
            mse_test = np.mean(np.square(test_data - reconstructed_test), axis=1)

            # labeling
            anomaly = np.where(mse_test > threshold_per_fold, 1, 0)
            anomaly_per_fold.append(anomaly)

        fold_ensemble = np.sum(anomaly_per_fold, axis=0)
        majority_vote = 3
        anomaly = np.where(fold_ensemble >= majority_vote, 1, 0)
        print(fold_ensemble)
        ensemble_results.extend(anomaly)

    # submission
    sample = pd.read_csv("./data/answer_sample.csv")
    sample["label"] = ensemble_results
    error_len = sum(ensemble_results)
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
