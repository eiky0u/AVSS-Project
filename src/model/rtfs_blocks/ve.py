from src.lipreading_pretrained.build_model import get_lipreading_model
import torch.nn as nn
import torch

class video_encoder(nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super().__init__()

        self.core = get_lipreading_model(
             "src/lipreading_pretrained/config/lrw_resnet18_dctcn.json", 
             pretrained=pretrained)
        
        if freeze:
            for param in self.core.parameters():
                param.requires_grad = False
                self.core.eval()

    def forward(self, x, lengths):
            #lengths = [T]*B
            return self.core(x, lengths=lengths).permute(0,2,1)  # [B, C, T]