import torch.nn as nn

#encoded images as input
class EncodedClassifier(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)
    
    
    
#control classifier uses images as input
class ControlClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(224*224, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = x.flatten(1)
        return self.classifier(x)