import torch
from base import BaseLitModule


class LitGANTrainer(BaseLitModule):
    def __init__(
        self,
        model,
        criterion_gen,
        criterion_disc,
        gen_optimizer_class,
        disc_optimizer_class,
        config,
        alpha,
    ):
        super().__init__()
        self.generator = model.generator
        self.discriminator = model.discriminator
        self.criterion_gen = criterion_gen
        self.criterion_disc = criterion_disc
        self.gen_optimizer_class = gen_optimizer_class
        self.disc_optimizer_class = disc_optimizer_class
        self.config = config
        self.alpha = alpha

    def forward(self, x):
        return self.generator(x)

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

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = batch

        if optimizer_idx == 0:
            generated_inputs = self.generator(inputs)
            g_loss = self._generator_loss(generated_inputs, inputs)
            self.log(
                "train_generator_loss",
                g_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            return g_loss

        elif optimizer_idx == 1:
            generated_inputs = self.generator(inputs)
            d_loss = self._discriminator_loss(inputs, generated_inputs)
            self.log(
                "train_discriminator_loss",
                d_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            return d_loss

    def validation_step(self, batch, batch_idx):
        inputs = batch
        batch_size = inputs.size(0)

        with torch.no_grad():
            generated_inputs = self.generator(inputs)
            disc_loss = self._discriminator_loss(inputs, generated_inputs)

            gen_loss = self._generator_loss(generated_inputs, inputs)

        return {"val_discriminator_loss": disc_loss, "val_generator_loss": gen_loss}

    def validation_epoch_end(self, outputs):
        avg_disc_loss = torch.stack(
            [x["val_discriminator_loss"] for x in outputs]
        ).mean()
        avg_gen_loss = torch.stack([x["val_generator_loss"] for x in outputs]).mean()

        self.log("val_discriminator_loss", avg_disc_loss, on_epoch=True, prog_bar=True)
        self.log("val_generator_loss", avg_gen_loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer_g = self.gen_optimizer_class(
            self.generator.parameters(), **self.config["gen_optimizer"]["args"]
        )
        optimizer_d = self.disc_optimizer_class(
            self.discriminator.parameters(), **self.config["disc_optimizer"]["args"]
        )

        return [optimizer_g, optimizer_d]
