#import
from src.project_parameters import ProjectParameters
from DeepLearningTemplate.model import BaseModel, load_from_checkpoint
import torch.nn as nn
import math
import torch.nn.functional as F


#class
class Generator(nn.Module):
    def __init__(self, in_chans, hidden_chans, chans_scale, depth) -> None:
        super().__init__()
        self.kernel_size = 8
        self.stride = 4
        self.depth = depth
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for index in range(self.depth):
            encode = []
            encode += [
                nn.Conv1d(in_channels=in_chans,
                          out_channels=hidden_chans,
                          kernel_size=self.kernel_size,
                          stride=self.stride),
                nn.ReLU(),
                nn.Conv1d(in_channels=hidden_chans,
                          out_channels=hidden_chans * chans_scale,
                          kernel_size=1),
                nn.GLU(1)
            ]
            self.encoder.append(nn.Sequential(*encode))
            decode = []
            decode += [
                nn.Conv1d(in_channels=hidden_chans,
                          out_channels=chans_scale * hidden_chans,
                          kernel_size=1),
                nn.GLU(1),
                nn.ConvTranspose1d(in_channels=hidden_chans,
                                   out_channels=in_chans,
                                   kernel_size=self.kernel_size,
                                   stride=self.stride)
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            in_chans = hidden_chans
            hidden_chans = int(chans_scale * hidden_chans)

    def valid_length(self, length):
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        return length

    def forward(self, x):
        _, _, length = x.shape
        x = F.pad(x, (0, self.valid_length(length) - length))
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., :x.shape[-1]]
            x = decode(x)
        return x[..., :length]


class Discriminator(nn.Module):
    def __init__(self, in_chans, hidden_chans, chans_scale, depth) -> None:
        super().__init__()
        self.kernel_size = 8
        self.stride = 4
        self.depth = depth
        self.encoder = nn.ModuleList()
        for index in range(self.depth):
            encode = []
            encode += [
                nn.Conv1d(in_channels=in_chans,
                          out_channels=hidden_chans,
                          kernel_size=self.kernel_size,
                          stride=self.stride),
                nn.ReLU(),
                nn.Conv1d(in_channels=hidden_chans,
                          out_channels=hidden_chans * chans_scale,
                          kernel_size=1),
                nn.GLU(1)
            ]
            self.encoder.append(nn.Sequential(*encode))
            in_chans = hidden_chans
            hidden_chans = int(chans_scale * hidden_chans)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1), nn.Flatten(),
            nn.Linear(in_features=hidden_chans // 2, out_features=1))

    def forward(self, x):
        for encode in self.encoder:
            x = encode(x)
        return x, self.classifier(x)


class UnsupervisedModel(BaseModel):
    def __init__(self, optimizers_config, lr, lr_schedulers_config, in_chans,
                 generator_hidden_chans, generator_chans_scale,
                 generator_depth, discriminator_hidden_chans,
                 discriminator_chans_scale, discriminator_depth) -> None:
        super().__init__(optimizers_config=optimizers_config,
                         lr=lr,
                         lr_schedulers_config=lr_schedulers_config)
        self.generator_xy = Generator(in_chans=in_chans,
                                      hidden_chans=generator_hidden_chans,
                                      chans_scale=generator_chans_scale,
                                      depth=generator_depth)
        self.generator_yx = Generator(in_chans=in_chans,
                                      hidden_chans=generator_hidden_chans,
                                      chans_scale=generator_chans_scale,
                                      depth=generator_depth)
        self.discriminator_x = Discriminator(
            in_chans=in_chans,
            hidden_chans=discriminator_hidden_chans,
            chans_scale=discriminator_chans_scale,
            depth=discriminator_depth)
        self.discriminator_y = Discriminator(
            in_chans=in_chans,
            hidden_chans=discriminator_hidden_chans,
            chans_scale=discriminator_chans_scale,
            depth=discriminator_depth)


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    #
    model = UnsupervisedModel(optimizers_config=None,
                              lr=None,
                              lr_schedulers_config=None,
                              in_chans=1,
                              generator_hidden_chans=32,
                              generator_chans_scale=2,
                              generator_depth=6,
                              discriminator_hidden_chans=32,
                              discriminator_chans_scale=2,
                              discriminator_depth=6)
