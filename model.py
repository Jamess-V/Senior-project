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
        if model_name == 'densenet121': 
            self.backbone = models.densenet121(pretrained=pretrained) 
            in_features = self.backbone.classifier.in_features 
            self.backbone.classifier = nn.Linear(in_features, num_classes) 
        elif model_name == 'resnet50': 
            self.backbone = models.resnet50(pretrained=pretrained) 
            in_features = self.backbone.fc.in_features 
            self.backbone.fc = nn.Linear(in_features, num_classes) 
        elif model_name == 'efficientnet_b0': 
            self.backbone = models.efficientnet_b0(pretrained=pretrained) 
            in_features = self.backbone.classifier[1].in_features 
            self.backbone.classifier = nn.Sequential( 
                nn.Dropout(p=0.2), 
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