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

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = batch
        batch_size = inputs.size(0)

        # Train the discriminator
        if optimizer_idx == 1:
            optimizer_D = self.optimizers()[1]
            optimizer_D.zero_grad()

            # Train on real data
            real_output = self.discriminator(inputs)
            real_label = torch.ones(batch_size, 1, device=self.device)
            real_loss = self.criterion_disc(real_output, real_label)
            self.log("discriminator_real_loss", real_loss)

            # Train on fake data
            fake_inputs = self.generator(inputs)
            fake_output = self.discriminator(fake_inputs.detach())
            fake_label = torch.zeros(batch_size, 1, device=self.device)
            fake_loss = self.criterion_disc(fake_output, fake_label)
            self.log("discriminator_fake_loss", fake_loss)

            # Update the discriminator
            disc_loss = real_loss + fake_loss
            self.log("discriminator_loss", disc_loss)
            disc_loss.backward()
            optimizer_D.step()

            return {"loss": disc_loss}

        # Train the generator
        if optimizer_idx == 0:
            optimizer_G = self.optimizers()[0]
            optimizer_G.zero_grad()

            # Use the existing fake_inputs
            fake_inputs = self.generator(inputs)
            fake_output = self.discriminator(fake_inputs)

            # Update the generator
            real_label = torch.ones(
                batch_size, 1, device=self.device
            )  # Assign new real_label
            gen_loss = self.criterion_gen(
                fake_inputs, inputs
            ) * self.alpha + self.criterion_disc(fake_output, real_label) * (
                1 - self.alpha
            )
            self.log("generator_loss", gen_loss)
            gen_loss.backward()
            optimizer_G.step()

            return {"loss": gen_loss}

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            self.generator.eval()
            self.discriminator.eval()

            inputs = batch
            batch_size = inputs.size(0)

            # Calculate losses for the discriminator
            real_output = self.discriminator(inputs)
            real_label = torch.ones(batch_size, 1, device=self.device)
            real_loss = self.criterion_disc(real_output, real_label)

            fake_inputs = self.generator(inputs)
            fake_output = self.discriminator(fake_inputs)
            fake_label = torch.zeros(batch_size, 1, device=self.device)
            fake_loss = self.criterion_disc(fake_output, fake_label)

            disc_loss = real_loss + fake_loss
            self.log(
                "val_discriminator_loss",
                disc_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

            # Calculate loss for the generator
            gen_loss = self.criterion_gen(
                fake_inputs, inputs
            ) * self.alpha + self.criterion_disc(fake_output, real_label) * (
                1 - self.alpha
            )
            self.log(
                "val_generator_loss",
                gen_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

    def validation_epoch_end(self, outputs):
        pass

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
