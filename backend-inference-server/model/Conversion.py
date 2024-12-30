import torch
import torchvision.models as models
from model_architecture import TransformerForClassification

def main():
    """ 
    Convert to torchscript and save it
    """
    model_path = "model.pt"
    model = TransformerForClassification()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    traced_model  = torch.jit.trace(model)
    
    traced_model.save("traced_model.pt")
    
if __name__ == "__main__":
    main()