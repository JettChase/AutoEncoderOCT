import torch
import torch.nn as nn

#error with latent_dim not actually 256
class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(AutoEncoder, self).__init__()
        
        #Encoder
        self.encoder = nn.Sequential(
            #add strides to downsample
            
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
        
        #Decoder
        self.decoder = nn.Sequential(
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
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed