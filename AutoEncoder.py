import torch.nn as nn

class Encoder(nn.Sequential):
    def __init__(self, latent_dim):
            super(Encoder, self).__init__(
                 #1x224x224
                nn.Conv2d(1, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                #32x112x112
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                #64x56x56
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU(),
                #128x28x28
                nn.Conv2d(128, latent_dim, 3, stride=2, padding=1),
                nn.ReLU(),
                #256x14x14
                nn.Flatten(),
                nn.Linear(256*14*14, latent_dim)
            )

class Decoder(nn.Sequential):
    def __init__(self, latent_dim):
            super(Decoder, self).__init__(
                #256x14x14
                nn.Linear(latent_dim, 256*14*14),
                nn.Unflatten(1, (256, 14, 14)),
                #128x28x28
                nn.ConvTranspose2d(latent_dim, 128, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                #64x56x56
                nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                #32x112x112
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                #1x224x224
                nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),    
                nn.Tanh() 
            )

class AutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed