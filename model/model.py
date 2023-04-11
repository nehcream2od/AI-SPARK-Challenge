import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from base import BaseLitModel
from .scheduler import CosineAnnealingWarmUpRestarts


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], hidden_size[0])
        self.fc4 = nn.Linear(hidden_size[0], output_size)
        self.layer_norm_hidden_0 = nn.LayerNorm(hidden_size[0])
        self.layer_norm_hidden_1 = nn.LayerNorm(hidden_size[1])
        self.rrelu = nn.RReLU()
        self.drop = nn.Dropout(0.0)

        self.apply(self._init_weights)

    def forward(self, x):
        x = self.rrelu(self.fc1(x))
        x = self.drop(x)
        x = self.layer_norm_hidden_0(x)
        x = self.rrelu(self.fc2(x))
        x = self.drop(x)
        x = self.layer_norm_hidden_1(x)
        x = self.rrelu(self.fc3(x))
        x = self.drop(x)
        x = self.layer_norm_hidden_0(x)
        x = self.fc4(x)
        return x

    def _init_weights(self, module):
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], output_size)
        self.layer_norm_hidden_0 = nn.LayerNorm(hidden_size[0])
        self.rrelu = nn.RReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(0.0)

        self.apply(self._init_weights)

    def forward(self, x):
        x = self.rrelu(self.fc1(x))
        x = self.drop(x)
        x = self.layer_norm_hidden_0(x)
        x = self.fc2(x)
        # x = self.sigmoid(x)
        return x

    def _init_weights(self, module):
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)


class LitGANModel(BaseLitModel):
    def __init__(
        self,
        input_size,
        output_size,
        gen_hidden_size,
        disc_hidden_size,
        criterion_gen,
        criterion_disc,
        gen_optimizer_class,
        disc_optimizer_class,
        gen_lr,
        disc_lr,
        config,
        alpha,
    ):
        super().__init__()
        self.config = config
        self.generator = Generator(input_size, gen_hidden_size, output_size)
        self.discriminator = Discriminator(output_size, disc_hidden_size, 1)
        self.criterion_gen = criterion_gen
        self.criterion_disc = criterion_disc
        self.gen_optimizer_class = gen_optimizer_class
        self.disc_optimizer_class = disc_optimizer_class
        self.gen_lr = gen_lr
        self.disc_lr = disc_lr
        self.alpha = alpha
        self.automatic_optimization = False
        self.training_outputs = []
        self.validation_outputs = []
        self.valid_mse = torchmetrics.MeanSquaredError()

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx):
        inputs = batch
        opt_gen, opt_disc = self.optimizers()

        # Generator training
        generated_inputs = self.generator(inputs)
        g_loss = self._generator_loss(generated_inputs, inputs)

        # Generator optimization
        self.manual_backward(g_loss)
        opt_gen.step()
        opt_gen.zero_grad()

        # Discriminator training
        generated_inputs = self.generator(inputs)
        d_loss = self._discriminator_loss(inputs, generated_inputs)

        # Discriminator optimization
        self.manual_backward(d_loss)
        opt_disc.step()
        opt_disc.zero_grad()

        self.training_outputs.append(
            {"train_discriminator_loss": d_loss, "train_generator_loss": g_loss}
        )

    def on_train_epoch_start(self):
        self.training_outputs = []

    def on_train_epoch_end(self):
        avg_disc_loss = torch.stack(
            [x["train_discriminator_loss"] for x in self.training_outputs]
        ).mean()
        avg_gen_loss = torch.stack(
            [x["train_generator_loss"] for x in self.training_outputs]
        ).mean()

        self.log(
            "train_discriminator_loss", avg_disc_loss, on_epoch=True, prog_bar=True
        )
        self.log("train_generator_loss", avg_gen_loss, on_epoch=True, prog_bar=True)

        # step scheduler
        self.lr_schedulers()[0].step(self.current_epoch)
        self.lr_schedulers()[1].step(self.current_epoch)

    def validation_step(self, batch, batch_idx):
        inputs = batch

        with torch.no_grad():
            generated_inputs = self.generator(inputs)
            disc_loss = self._discriminator_loss(inputs, generated_inputs)

            gen_loss = self._generator_loss(generated_inputs, inputs)

        self.validation_outputs.append(
            {"val_discriminator_loss": disc_loss, "val_generator_loss": gen_loss}
        )
        self.valid_mse.update(self(inputs), inputs)

    def on_validation_epoch_start(self):
        self.validation_outputs = []

    def on_validation_epoch_end(self):
        avg_disc_loss = torch.stack(
            [x["val_discriminator_loss"] for x in self.validation_outputs]
        ).mean()
        avg_gen_loss = torch.stack(
            [x["val_generator_loss"] for x in self.validation_outputs]
        ).mean()

        self.log("val_discriminator_loss", avg_disc_loss, on_epoch=True, prog_bar=True)
        self.log("val_generator_loss", avg_gen_loss, on_epoch=True, prog_bar=True)
        self.log("val_mse", self.valid_mse.compute(), on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer_g = self.gen_optimizer_class(
            self.generator.parameters(), lr=self.gen_lr
        )
        optimizer_d = self.disc_optimizer_class(
            self.discriminator.parameters(), lr=self.disc_lr
        )

        scheduler_g = CosineAnnealingWarmUpRestarts(
            optimizer_g,
            T_0=self.config["gen_scheduler"]["args"]["T_0"],
            T_mult=self.config["gen_scheduler"]["args"]["T_mult"],
            eta_max=self.config["gen_scheduler"]["args"]["eta_max"],
            T_up=self.config["gen_scheduler"]["args"]["T_up"],
            gamma=self.config["gen_scheduler"]["args"]["gamma"],
        )
        scheduler_d = CosineAnnealingWarmUpRestarts(
            optimizer_d,
            T_0=self.config["disc_scheduler"]["args"]["T_0"],
            T_mult=self.config["disc_scheduler"]["args"]["T_mult"],
            eta_max=self.config["disc_scheduler"]["args"]["eta_max"],
            T_up=self.config["disc_scheduler"]["args"]["T_up"],
            gamma=self.config["disc_scheduler"]["args"]["gamma"],
        )

        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]

    def _generator_loss(self, generated_inputs, inputs):
        valid = torch.ones(inputs.shape[0], 1, device=self.device)
        return self.criterion_gen(
            generated_inputs, inputs
        ) * self.alpha + self.criterion_disc(
            self.discriminator(generated_inputs), valid
        ) * (
            1 - self.alpha
        )

    def _discriminator_loss(self, real_inputs, fake_inputs):
        valid = torch.ones(real_inputs.shape[0], 1, device=self.device)
        fake = torch.zeros(real_inputs.shape[0], 1, device=self.device)
        return self.criterion_disc(
            self.discriminator(real_inputs), valid
        ) + self.criterion_disc(self.discriminator(fake_inputs.detach()), fake)


class GeneratorWrapper(BaseLitModel):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, x):
        return self.generator(x)
