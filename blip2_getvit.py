import sys

import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
from torchinfo import summary
import matplotlib.pyplot as plt
import numpy as np

from torchinfo import summary

def show_vis_processor(vis_processor_output, save_name='image.png'):
    data = vis_processor_output.cpu().numpy()
    data = (data - data.min()) / (data.max() - data.min())
    data = np.transpose(data[0], (1, 2, 0))
    plt.imshow(data)
    plt.axis('off')  # 軸を表示しない
    plt.savefig(save_name)
    plt.close()

# check GPU usage
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count()==0: print('Use 1 GPU')
else: print(f'Use {torch.cuda.device_count()} GPUs')

# モデルの読み込み
model, vis_processors, _ = load_model_and_preprocess(
    name='blip2_t5',
    model_type='pretrain_flant5xxl',
    is_eval=True,
    device=device
)

vit, ln_vision = model.get_vision_encoder()

vit = vit.to(torch.float32)
ln_vision = ln_vision.to(torch.float32)

summary(vit)
#

raw_image = Image.open('/taiga/experiment/BLIP-2/image/merlion.png').convert('RGB')
image = vis_processors['eval'](raw_image).unsqueeze(0).to(device)

print('input_data: ', image.dtype)
print('input_data: ', image.shape)


output = vit(x=image.to(torch.float32))

#output = ln_vision(output)

print('vit_output : ', output)
print('vit_output.shape : ', output.shape)
