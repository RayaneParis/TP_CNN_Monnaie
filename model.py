import torch
import torch.nn as nn
import torchvision.models as models


class AlexNet(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
                # 3->96, kernel 11*11
                nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
                # 96 -> 256, kernel 5*5
                nn.Conv2d(96, 256, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
                # 256 -> 384, kernel 3*3
                nn.Conv2d(256, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                # 284 -> 384, kernel 3*3
                nn.Conv2d(384, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                # 384 -> 284, kernel 3*3
                nn.Conv2d(384, 284, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
        self.avgPool = nn.AdaptativeAvgPool2d((6, 6))

        # mettre en place le classificateur
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes), # en fonction du nombre de classes
        )

        self._initialize_weights()
    def forward(self, x):
        x = self.features(x)
        x = self.avgPool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constnat_(m.bias, 0)

def get_pretrain_alexnet(num_classes: int):
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)

    #for param in model.features.parameters():
    #    param.requires_grad = False

    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)

    return model


        

        