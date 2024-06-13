import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=10, init_weights=False):   
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(  #package
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=2, padding=1),  # input[3, 32, 32]
            nn.BatchNorm2d(256,affine=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  
                   
            nn.Conv2d(in_channels=256, out_channels=768, kernel_size=3, stride=1, padding=1),   
            nn.BatchNorm2d(768,affine=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),     
            
            nn.Conv2d(in_channels=768, out_channels=1536, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(1536,affine=False),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=1536, out_channels=1024, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(1024,affine=False),  
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1), 
            nn.BatchNorm2d(1024,affine=False),    
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), 
        )
        # For certified unlearning, devide the AlexNet.classifier into 4 parts.
        self.classifier1 = nn.Dropout()
        self.classifier2 = nn.Sequential(
            nn.Linear(1024 * 1 * 1, 8192),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(8192, 1024),
            nn.ReLU(inplace=True),
        )
        self.classifier4 = nn.Linear(1024, num_classes) 
        # For the sake of hessen matrix calculation, squeeze the size of classifiers.
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1) #flatten
        x = self.classifier4(self.classifier3(self.classifier2(self.classifier1((x)))))
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') 
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01) 
                nn.init.constant_(m.bias, 0)
    
                            