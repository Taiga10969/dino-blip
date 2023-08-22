import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FigureDataset(Dataset):
    def __init__(self, data_paths, transform=None):
        self.data_paths = data_paths
        self.transform = transform
        self.image_filenames = self._load_image_filenames()

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
        
        return image


if __name__=='__main__':

    data_paths = [#"/taiga/Datasets/scicap_data/SciCap-Yes-Subfig-Img/train", 
                  #"/taiga/Datasets/scicap_data/SciCap-Yes-Subfig-Img/test", 
                  #"/taiga/Datasets/scicap_data/SciCap-No-Subfig-Img/train", 
                  "/taiga/Datasets/scicap_data/SciCap-No-Subfig-Img/test", 
                  #"/taiga/Datasets/moonshot-dataset/figures",
                  ]

    dataset = FigureDataset(data_paths)

    print('len(dataset) : ', len(dataset))
