import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FigureDataset(Dataset):
    def __init__(self, data_paths, transform=None, vis_processors=None):
        self.data_paths = data_paths
        self.transform = transform
        self.image_filenames = self._load_image_filenames()
        self.vis_processors = vis_processors

    def _load_image_filenames(self):
        image_filenames = []
        for data_path in self.data_paths:
            filenames = os.listdir(data_path)
            image_filenames.extend([os.path.join(data_path, filename) for filename in filenames])
        return image_filenames

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)
        
        if self.vis_processors:
            image = self.vis_processors['eval'](image).unsqueeze(0)
        
        return image


if __name__=='__main__':
    import torch
    from lavis.models import load_model_and_preprocess

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


    data_paths = [#"/taiga/Datasets/scicap_data/SciCap-Yes-Subfig-Img/train", 
                  "/taiga/Datasets/scicap_data/SciCap-Yes-Subfig-Img/test", 
                  #"/taiga/Datasets/scicap_data/SciCap-No-Subfig-Img/train", 
                  "/taiga/Datasets/scicap_data/SciCap-No-Subfig-Img/test", 
                  #"/taiga/Datasets/moonshot-dataset/figures",
                  ]

    dataset = FigureDataset(data_paths=data_paths, vis_processors=vis_processors)

    print('len(dataset) : ', len(dataset))

    image = dataset[0]

    print(image)
