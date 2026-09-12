"""Model definition with pre-trained backbone and MC Dropout support.""" 

import torch 
import torch.nn as nn 
import torchvision.models as models 

class MedicalAIModel(nn.Module): 
    """Pre-trained CNN with MC Dropout support."""      

    def __init__(self, model_name='densenet121', num_classes=1, pretrained=True, mc_dropout_keep_prob=0.5): 
        super().__init__()        

        self.model_name = model_name 
        self.mc_dropout_keep_prob = mc_dropout_keep_prob          

        # Load pre-trained model 
        dropout_probability = 1.0 - mc_dropout_keep_prob
        if not 0.0 < mc_dropout_keep_prob <= 1.0:
            raise ValueError("mc_dropout_keep_prob must be in the (0, 1] range")

        if model_name == 'densenet121':
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            self.backbone = models.densenet121(weights=weights)
            in_features = self.backbone.classifier.in_features 
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_probability),
                nn.Linear(in_features, num_classes),
            )
        elif model_name == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features 
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_probability),
                nn.Linear(in_features, num_classes),
            )
        elif model_name == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features 
            self.backbone.classifier = nn.Sequential( 
                nn.Dropout(p=dropout_probability),
                nn.Linear(in_features, num_classes) 
            ) 
        else: 
            raise ValueError(f"Unsupported model: {model_name}") 
         
        # Store dropout layers for MC Dropout 
        self.dropout_layers = [] 
        self._collect_dropout_layers(self.backbone) 
         
    def _collect_dropout_layers(self, module): 
        """Recursively collect all Dropout layers for MC Dropout.""" 
        for name, child in module.named_children(): 
            if isinstance(child, nn.Dropout): 
                self.dropout_layers.append(child) 
            else: 
                self._collect_dropout_layers(child) 
     
    def forward(self, x): 
        """Forward pass with dropout enabled.""" 
        return self.backbone(x) 

    def enable_mc_dropout(self): 
        """Enable dropout for MC Dropout inference.""" 
        for module in self.modules(): 
            if isinstance(module, nn.Dropout): 
                module.train()  # Keep in training mode for dropout to be active 

    def disable_mc_dropout(self): 
        """Disable dropout for standard inference.""" 
        for module in self.modules(): 
            if isinstance(module, nn.Dropout): 
                module.eval()  # Switch to evaluation mode 

def create_model(config): 
    """Create model with configuration.""" 
    model = MedicalAIModel( 
        model_name=config.model_name, 
        num_classes=config.num_classes, 
        pretrained=config.pretrained, 
        mc_dropout_keep_prob=config.mc_dropout_keep_prob 
    ) 
    return model.to(config.device) 