import torch
import torchmetrics
from base import BaseLitModule


class LitGANTrainer(BaseLitModule):
    def __init__(
        self,
        model,
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
        self.generator = model.generator
        self.discriminator = model.discriminator
        self.criterion_gen = criterion_gen
        self.criterion_disc = criterion_disc
        self.gen_optimizer_class = gen_optimizer_class
        self.disc_optimizer_class = disc_optimizer_class
        self.gen_lr = gen_lr
        self.disc_lr = disc_lr
        self.alpha = alpha
        self.automatic_optimization = False
        self.training_ouputs = []
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

    def validation_step(self, batch, batch_idx):
        inputs = batch
        batch_size = inputs.size(0)

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

        return [optimizer_g, optimizer_d]

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
