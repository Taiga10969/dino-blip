import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from lavis.models import load_model_and_preprocess
from lavis.models.eva_vit import create_eva_vit_g, get_blip2_vit_g
from PIL import Image
from torchinfo import summary
import matplotlib.pyplot as plt
import numpy as np


from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import pickle

import umap
import tqdm

import dataset

from torchinfo import summary

# check GPU usage
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count()==0: print('Use 1 GPU')
else: print(f'Use {torch.cuda.device_count()} GPUs')

# モデルの読み込み
_, vis_processors, _ = load_model_and_preprocess(
    name='blip2_t5',
    model_type='pretrain_flant5xxl',
    is_eval=True,
    device=device
)

#vit, ln_vision = model.get_vision_encoder()
vit = create_eva_vit_g()
#vit = get_blip2_vit_g()

vit = nn.DataParallel(vit.to(device, torch.float32))


data_paths = [#"/taiga/Datasets/scicap_data/SciCap-Yes-Subfig-Img/train", 
              #"/taiga/Datasets/scicap_data/SciCap-Yes-Subfig-Img/test", 
              #"/taiga/Datasets/scicap_data/SciCap-No-Subfig-Img/train", 
              "/taiga/Datasets/scicap_data/SciCap-No-Subfig-Img/test", 
              #"/taiga/Datasets/moonshot-dataset/figures",
              #"/taiga/Datasets/umap_dataset",
              "/taiga/Datasets/COCO_dataset2014/val"
              ]

datasets = dataset.FigureDataset(data_paths=data_paths, vis_processors=vis_processors)
dataloader = DataLoader(datasets, batch_size=32, shuffle=False, num_workers=4, drop_last=False)

dim = 1408

# 特徴量を保存するためのリスト
features = []

for batch_index, inputs in enumerate(tqdm.tqdm(dataloader)):
    #print(inputs.size())
    b, _, _, _ = inputs.size()
    outputs = vit(inputs.to(device, torch.float32))
       
    # 特徴量をリストに追加
    features.append(outputs.squeeze().cpu().detach().numpy())
    #print('features.shape : ', np.shape(features))

## pickleで保存（書き出し）
#with open('features.pickle', mode='wb') as fo:
#  pickle.dump(features, fo)
#
# 特徴量を1つの配列に結合
all_features = np.concatenate(features, axis=0)
#
## pickleで保存（書き出し）
#with open('all_features.pickle', mode='wb') as fo:
#  pickle.dump(all_features, fo)


reshaped_features = all_features[:, 0, :]

print('reshaped_features.shape', reshaped_features.shape) #(28327, 257, 1408)

# UMAPで次元圧縮
reducer = umap.UMAP()
umap_embedding = reducer.fit_transform(reshaped_features)

# 可視化
#plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], s=5)
#plt.title("UMAP Projection of reshaped_features")

datasets = dataset.umap_Dataset(data_paths=data_paths)

plt.figure(figsize=(10, 8))
draw_image2d("output.png", umap_embedding[:5000], datasets[:5000],zoom=0.05, fig_size=(10, 8), dpi=1000)



