from utils import *
from torchvision.models import resnet18
from collections import OrderedDict 

def create_model(num_classes,size):
    """
    Function. Builds a resnet18-based model. 
    It also modifies the Linear layer according to the number of classes in the problem.

    Args. num_classes(Int). 

    """
    model = resnet18(weights='DEFAULT')
    #model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

#Model to another file
class Ariel_model(nn.Module):
    """
    Defines a custom model.
    """
    def __init__(self, num_classes,image_size, output_layers, *args):
        super().__init__()
        self.model = create_model(num_classes,image_size)
        self.fhooks = []
        self.output_layers = output_layers
        self.selected_out = OrderedDict()

        for nam, mod in enumerate(list(self.model._modules.keys())):
            if nam in self.output_layers:
                self.fhooks.append(getattr(self.model,mod).register_forward_hook(self.forward_hook(mod)))

    def forward(self, x):
        out = self.model(x)
        return torch.nn.functional.softmax(out, dim=1), self.selected_out
    
    def forward_hook(self,layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook