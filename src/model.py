#import
from pandas import options
from src.project_parameters import ProjectParameters
from DeepLearningTemplate.model import BaseModel, load_from_checkpoint
import torch.nn as nn
import math
import torch.nn.functional as F
import torch
from os.path import isfile
from torchsummary import summary


#def
def create_model(project_parameters):
    model = UnsupervisedModel(
        optimizers_config=project_parameters.optimizers_config,
        lr=project_parameters.lr,
        lr_schedulers_config=project_parameters.lr_schedulers_config,
        in_chans=project_parameters.in_chans,
        generator_hidden_chans=project_parameters.generator_hidden_chans,
        generator_chans_scale=project_parameters.generator_chans_scale,
        generator_depth=project_parameters.generator_depth,
        discriminator_hidden_chans=project_parameters.discriminator_hidden_chans,
        discriminator_chans_scale=project_parameters.discriminator_chans_scale,
        discriminator_depth=project_parameters.discriminator_depth)
    if project_parameters.checkpoint_path is not None:
        if isfile(project_parameters.checkpoint_path):
            model = load_from_checkpoint(
                device=project_parameters.device,
                checkpoint_path=project_parameters.checkpoint_path,
                model=model)
        else:
            assert False, 'please check the checkpoint_path argument.\nthe checkpoint_path value is {}.'.format(
                project_parameters.checkpoint_path)
    return model


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
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def configure_optimizers(self):
        optimizers_g_xy = self.parse_optimizers(
            params=self.generator_xy.parameters())
        optimizers_g_yx = self.parse_optimizers(
            params=self.generator_yx.parameters())
        optimizers_d_x = self.parse_optimizers(
            params=self.discriminator_x.parameters())
        optimizers_d_y = self.parse_optimizers(
            params=self.discriminator_y.parameters())
        if self.lr_schedulers_config is not None:
            lr_schedulers_g_xy = self.parse_lr_schedulers(
                optimizers=optimizers_g_xy)
            lr_schedulers_g_yx = self.parse_lr_schedulers(
                optimizers=optimizers_g_yx)
            lr_schedulers_d_x = self.parse_lr_schedulers(
                optimizers=optimizers_d_x)
            lr_schedulers_d_y = self.parse_lr_schedulers(
                optimizers=optimizers_d_y)
            return [
                optimizers_g_xy[0], optimizers_g_yx[0], optimizers_d_x[0],
                optimizers_d_y[0]
            ], [
                lr_schedulers_g_xy[0], lr_schedulers_g_yx[0],
                lr_schedulers_d_x[0], lr_schedulers_d_y[0]
            ]
        else:
            return [
                optimizers_g_xy[0], optimizers_g_yx[0], optimizers_d_x[0],
                optimizers_d_y[0]
            ]

    def weights_init(self, module):
        classname = module.__class__.__name__
        if classname.find('Conv') != -1:
            module.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            module.weight.data.normal_(1.0, 0.02)
            module.bias.data.fill_(0)

    def forward(self, x):
        y_hat = self.generator_xy(x)
        x_hat = self.generator_yx(y_hat)
        return y_hat, x_hat

    def shared_step(self, batch):
        x, y = batch
        # x -> x_y -> x_y_x
        x_y = self.generator_xy(x)
        x_y_x = self.generator_yx(x_y)
        # y _> y_x -> y_x_y
        y_x = self.generator_yx(y)
        y_x_y = self.generator_xy(y_x)
        return x, y, x_y, x_y_x, y_x, y_x_y

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y, x_y, x_y_x, y_x, y_x_y = self.shared_step(batch=batch)

        if optimizer_idx == 0 or optimizer_idx == 1:
            #adversarial loss generator y
            feat_y, proba_y = self.discriminator_y(y)
            feat_y_hat, proba_y_hat = self.discriminator_y(x_y)
            g_loss_feat_y = self.l1_loss(feat_y_hat, feat_y)
            g_loss_proba_y = self.bce_loss(proba_y_hat,
                                           torch.ones_like(input=proba_y_hat))

            #adversarial loss generator x
            feat_x, proba_x = self.discriminator_x(x)
            feat_x_hat, proba_x_hat = self.discriminator_x(y_x)
            g_loss_feat_x = self.l1_loss(feat_x_hat, feat_x)
            g_loss_proba_x = self.bce_loss(proba_x_hat,
                                           torch.ones_like(input=proba_x_hat))

            #cycle consistency loss
            g_loss_x_y_x = self.l1_loss(x_y_x, x)
            g_loss_y_x_y = self.l1_loss(y_x_y, y)
            g_loss = g_loss_feat_y + g_loss_proba_y + g_loss_feat_x + g_loss_proba_x + g_loss_x_y_x + g_loss_y_x_y

            self.log('train_loss',
                     g_loss,
                     on_step=True,
                     on_epoch=True,
                     prog_bar=True,
                     logger=True)
            self.log('train_loss_generator',
                     g_loss,
                     on_step=True,
                     on_epoch=True,
                     prog_bar=True,
                     logger=True)
            return g_loss

        if optimizer_idx == 2 or optimizer_idx == 3:
            #adversarial loss discriminator y
            feat_y, proba_y = self.discriminator_y(y)
            feat_y_hat, proba_y_hat = self.discriminator_y(x_y.detach())
            d_loss_proba_y = self.bce_loss(proba_y,
                                           torch.ones_like(input=proba_y))
            d_loss_proba_y_hat = self.bce_loss(
                proba_y_hat, torch.zeros_like(input=proba_y_hat))

            #adversarial loss discriminator x
            feat_x, proba_x = self.discriminator_x(x)
            feat_x_hat, proba_x_hat = self.discriminator_x(y_x.detach())
            d_loss_proba_x = self.bce_loss(proba_x,
                                           torch.ones_like(input=proba_x))
            d_loss_proba_x_hat = self.bce_loss(
                proba_x_hat, torch.zeros_like(input=proba_x_hat))

            d_loss = d_loss_proba_y + d_loss_proba_y_hat + d_loss_proba_x + d_loss_proba_x_hat

            self.log('train_loss_discriminator',
                     d_loss,
                     on_step=True,
                     on_epoch=True,
                     prog_bar=True,
                     logger=True)
            return d_loss

    def validation_step(self, batch, batch_idx):
        x, y, x_y, x_y_x, y_x, y_x_y = self.shared_step(batch=batch)
        #adversarial loss generator y
        feat_y, proba_y = self.discriminator_y(y)
        feat_y_hat, proba_y_hat = self.discriminator_y(x_y)
        g_loss_feat_y = self.l1_loss(feat_y_hat, feat_y)
        g_loss_proba_y = self.bce_loss(proba_y_hat,
                                       torch.ones_like(input=proba_y_hat))

        #adversarial loss generator x
        feat_x, proba_x = self.discriminator_x(x)
        feat_x_hat, proba_x_hat = self.discriminator_x(y_x)
        g_loss_feat_x = self.l1_loss(feat_x_hat, feat_x)
        g_loss_proba_x = self.bce_loss(proba_x_hat,
                                       torch.ones_like(input=proba_x_hat))

        #cycle consistency loss
        g_loss_x_y_x = self.l1_loss(x_y_x, x)
        g_loss_y_x_y = self.l1_loss(y_x_y, y)
        g_loss = g_loss_feat_y + g_loss_proba_y + g_loss_feat_x + g_loss_proba_x + g_loss_x_y_x + g_loss_y_x_y

        #adversarial loss discriminator y
        feat_y, proba_y = self.discriminator_y(y)
        feat_y_hat, proba_y_hat = self.discriminator_y(x_y.detach())
        d_loss_proba_y = self.bce_loss(proba_y, torch.ones_like(input=proba_y))
        d_loss_proba_y_hat = self.bce_loss(proba_y_hat,
                                           torch.zeros_like(input=proba_y_hat))

        #adversarial loss discriminator x
        feat_x, proba_x = self.discriminator_x(x)
        feat_x_hat, proba_x_hat = self.discriminator_x(y_x.detach())
        d_loss_proba_x = self.bce_loss(proba_x, torch.ones_like(input=proba_x))
        d_loss_proba_x_hat = self.bce_loss(proba_x_hat,
                                           torch.zeros_like(input=proba_x_hat))

        d_loss = d_loss_proba_y + d_loss_proba_y_hat + d_loss_proba_x + d_loss_proba_x_hat

        if d_loss.item() < 1e-5:
            self.discriminator_x.apply(self.weights_init)
            self.discriminator_y.apply(self.weights_init)

        self.log('val_loss', g_loss)
        self.log('val_loss_generator', g_loss, prog_bar=True)
        self.log('val_loss_discriminator', d_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y, x_y, x_y_x, y_x, y_x_y = self.shared_step(batch=batch)
        #adversarial loss generator y
        feat_y, proba_y = self.discriminator_y(y)
        feat_y_hat, proba_y_hat = self.discriminator_y(x_y)
        g_loss_feat_y = self.l1_loss(feat_y_hat, feat_y)
        g_loss_proba_y = self.bce_loss(proba_y_hat,
                                       torch.ones_like(input=proba_y_hat))

        #adversarial loss generator x
        feat_x, proba_x = self.discriminator_x(x)
        feat_x_hat, proba_x_hat = self.discriminator_x(y_x)
        g_loss_feat_x = self.l1_loss(feat_x_hat, feat_x)
        g_loss_proba_x = self.bce_loss(proba_x_hat,
                                       torch.ones_like(input=proba_x_hat))

        #cycle consistency loss
        g_loss_x_y_x = self.l1_loss(x_y_x, x)
        g_loss_y_x_y = self.l1_loss(y_x_y, y)
        g_loss = g_loss_feat_y + g_loss_proba_y + g_loss_feat_x + g_loss_proba_x + g_loss_x_y_x + g_loss_y_x_y

        self.log('test_loss', g_loss)


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # create model
    model = create_model(project_parameters=project_parameters)

    # display model information
    summary(model=model,
            input_size=(project_parameters.in_chans,
                        project_parameters.sample_rate),
            device='cpu')

    # create input data
    x = torch.rand(project_parameters.batch_size, project_parameters.in_chans,
                   project_parameters.sample_rate)

    # get model output
    y_hat, x_hat = model(x)

    # display the dimension of input and output
    print('the dimension of input: {}'.format(x.shape))
    print('the dimension of y_hat: {}'.format(y_hat.shape))
    print('the dimension of x_hat: {}'.format(x_hat.shape))
