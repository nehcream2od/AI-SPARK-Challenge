import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


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


class GANModel(BaseModel):
    def __init__(self, input_size, output_size, gen_hidden_size, dsc_hidden_size):
        super(GANModel, self).__init__()
        self.generator = Generator(input_size, gen_hidden_size, output_size)
        self.discriminator = Discriminator(output_size, dsc_hidden_size, 1)


class GeneratorWrapper(BaseModel):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, x):
        return self.generator(x)
