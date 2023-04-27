import torch
import torchvision.transforms as T
import torch.nn as nn
from ffcGenerator import *
from ffcGenerator_no_att import FFC_generator as FFC_generator_raw
from PIL import Image
import cv2

class DehazingModel(nn.Module):
    def __init__(self, path, use_attention='True', ngf=32) -> None:
        super().__init__()
        self.device = torch.device("cuda:" + str(0) if (torch.cuda.is_available() and int(0) >= 0) else "cpu")  
             
        self.state_dict = torch.load(path, map_location=self.device)
        if use_attention:
            self.model = FFC_generator(use_attention=use_attention, ngf=ngf)
        else:
            self.model = FFC_generator_raw(use_attention=use_attention, ngf=ngf)
            
        self.model.load_state_dict(self.state_dict)
        self.model.eval()
        
    def forward(self, x):
        return self.model(x)[0]
        
if __name__ == '__main__':
    #without attention
    use_attention = False
    if use_attention:
        path = "models/17_model_G.pth"
        ngf = 32
    else:
        path = "models/100_model_G_not_att.pth"
        ngf = 16
    #with attention
    device = torch.device("cuda:" + str(0) if (torch.cuda.is_available() and int(0) >= 0) else "cpu")  
    model = DehazingModel(path, use_attention=use_attention, ngf=ngf).to(device)
    img = Image.open("input/received.png")
    transform = T.ToTensor()
    img = transform(img).unsqueeze(0).to(device)
    #x = torch.zeros((1,3,256,256)).to(device)
    out, transm_map = model(img)
    out_np = out.detach().squeeze(0).permute(1,2,0).cpu().numpy()
    transm_np = transm_map.detach().squeeze(0).squeeze(0).cpu().numpy()
    out_np = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite("output/out_img.png", out_np*255)
    cv2.imwrite("output/transm.png", transm_np*255)