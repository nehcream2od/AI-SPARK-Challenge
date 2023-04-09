import torch
from base import BaseLitModule
from model.optimizer import Lion


class LitGANTrainer(BaseLitModule):
    def __init__(
        self,
        model,
        criterion_gen,
        criterion_disc,
        config,
        alpha=0.5,
    ):
        super().__init__()
        self.generator = model.generator
        self.discriminator = model.discriminator
        self.criterion_gen = criterion_gen
        self.criterion_disc = criterion_disc
        self.config = config
        self.alpha = alpha
        self.automatic_optimization = False

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx):
        inputs = batch
        batch_size = inputs.shape[0]

        optimizer_g, optimizer_d = self.optimizers()

        # Train generator
        self.toggle_optimizer(optimizer_g, 0)
        generated_inputs = self.generator(inputs)
        valid = torch.ones(batch_size, 1, device=self.device)

        g_loss = self.criterion_gen(
            generated_inputs, inputs
        ) * self.alpha + self.criterion_disc(
            self.discriminator(generated_inputs), valid
        ) * (
            1 - self.alpha
        )

        self.log(
            "train_generator_loss", g_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # Train discriminator
        self.toggle_optimizer(optimizer_d, 1)
        valid = torch.ones(batch_size, 1, device=self.device)
        real_loss = self.criterion_disc(self.discriminator(inputs), valid)
        fake = torch.zeros(batch_size, 1, device=self.device)
        fake_loss = self.criterion_disc(
            self.discriminator(generated_inputs.detach()), fake
        )

        d_loss = (real_loss + fake_loss) / 2
        self.log(
            "train_discriminator_loss",
            d_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def validation_step(self, batch, batch_idx):
        inputs = batch
        batch_size = inputs.size(0)

        with torch.no_grad():
            real_output = self.discriminator(inputs)
            valid = torch.ones(batch_size, 1, device=self.device)
            real_loss = self.criterion_disc(real_output, valid)

            generated_inputs = self.generator(inputs)
            fake_output = self.discriminator(generated_inputs)
            fake = torch.zeros(batch_size, 1, device=self.device)
            fake_loss = self.criterion_disc(fake_output, fake)

            disc_loss = real_loss + fake_loss

            gen_loss = self.criterion_gen(
                generated_inputs, inputs
            ) * self.alpha + self.criterion_disc(fake_output, valid) * (1 - self.alpha)

        return {"val_discriminator_loss": disc_loss, "val_generator_loss": gen_loss}

    def validation_epoch_end(self, outputs):
        avg_disc_loss = torch.stack(
            [x["val_discriminator_loss"] for x in outputs]
        ).mean()
        avg_gen_loss = torch.stack([x["val_generator_loss"] for x in outputs]).mean()

        self.log("val_discriminator_loss", avg_disc_loss, on_epoch=True, prog_bar=True)
        self.log("val_generator_loss", avg_gen_loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        if self.config["gen_optimizer"]["type"] == "Lion":
            optimizer_g = Lion(
                self.generator.parameters(), **self.config["gen_optimizer"]["args"]
            )
        else:
            optimizer_g = getattr(torch.optim, self.config["gen_optimizer"]["type"])(
                self.generator.parameters(), **self.config["gen_optimizer"]["args"]
            )
        if self.config["dsc_optimizer"]["type"] == "Lion":
            optimizer_d = Lion(
                self.discriminator.parameters(), **self.config["dsc_optimizer"]["args"]
            )
        else:
            optimizer_d = getattr(torch.optim, self.config["dsc_optimizer"]["type"])(
                self.discriminator.parameters(), **self.config["dsc_optimizer"]["args"]
            )
        return [optimizer_g, optimizer_d]
