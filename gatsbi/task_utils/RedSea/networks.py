"""
Architecture similar to Pix-to-pix GAN.
Code closely follows
https://machinelearningmastery.com/\
    how-to-develop-a-pix2pix-gan-for-image-to-image-translation/
"""
import torch
import torch.nn as nn

from gatsbi.networks import (AddConvNoise, BaseNetwork, Discriminator,
                             ModuleWrapper)


def final_layer(x):
    th1 = torch.exp(x[:,0:5])
    th2 = x[:,5].unsqueeze(1)
    th3 = torch.exp(x[:,6:])
    return torch.cat([th1,th2, th3],dim=1)

class ConvBlock(nn.Module):
    """Convolution block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=4,
        stride=1,
        padding=0,
        spec_norm=False,
        norm=False,
        nonlin=True,
    ):
        """Set up convolution block."""
        super(ConvBlock, self).__init__()
        block = []
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        if spec_norm:
            conv = nn.utils.spectral_norm(conv)
        block.append(conv)

        if nonlin:
            block.append(nn.LeakyReLU(0.2))

        if norm:
            batch_norm = nn.BatchNorm2d(out_channels)
            # if spec_norm:
            #     batch_norm = nn.utils.spectral_norm(batch_norm)
            block.append(batch_norm)

        self.block = nn.Sequential(*block)

    def forward(self, x):
        """Forward pass."""
        return self.block(x)


class TransConvBlock(nn.Module):
    """Transpose convolutional block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        spec_norm=False,
    ):
        """Set up transpose convolutional block."""

        super(TransConvBlock, self).__init__()

        conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        batch_norm = nn.BatchNorm2d(out_channels)
        if spec_norm:
            conv = nn.utils.spectral_norm(conv)

        self.block = nn.Sequential(conv, nn.ReLU(), batch_norm)

    def forward(self, x):
        """Forward pass."""
        return self.block(x)



class RedSeaGenerator(BaseNetwork):
    """Generator network for RedSea model."""

    def __init__(self):
        """Set up generator network."""
        gen_hidden_layers = [
            ConvBlock(1,16,kernel_size = (14,22)),
            ConvBlock(16, 32, 10),
            ConvBlock(32, 64, 5),
            ConvBlock(64, 128, 3),
            AddConvNoise(1, 200, 128, 1, heteroscedastic=True, conv2d=True, add=False),
            nn.Linear(256,500),
            nn.Linear(500,8),
        ]
        super(RedSeaGenerator, self).__init__(hidden_layers=gen_hidden_layers)

    def forward(self, x):
        """Forward pass."""
        #print("X SHAPE ", x.shape)
        enc1 = self._hidden_layers[0](x)
        #print("ENC1 SHAPE ", enc1.shape)
        enc2 = self._hidden_layers[1](enc1)
        #print("ENC2 SHAPE ", enc2.shape)
        enc3 = self._hidden_layers[2](enc2)
        #print("ENC3 SHAPE ", enc3.shape)

        latent = self._hidden_layers[3](enc3)
        #print("LATENT SHAPE ", latent.shape)
        noisy_latent = self._hidden_layers[4](latent)
        #print("N_LATENT SHAPE ", noisy_latent.shape)
        
        sq1 = torch.squeeze(noisy_latent, dim=2)
        sq2 = torch.squeeze(sq1, dim=2)
        #print("Squeezed shape ", sq2.shape)

        dec1 = self._hidden_layers[5](sq2)
        #print("dec1 SHAPE ", dec1.shape)
        dec2 = self._hidden_layers[6](dec1)
        #print("dec2 SHAPE ", dec2.shape)
        output = final_layer(dec2).unsqueeze(1).unsqueeze(1)
        #print("OUTPUT SHAPE ", output.shape)
        return output
