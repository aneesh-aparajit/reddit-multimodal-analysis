import os
from typing import Dict

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from config import Config


class MemotionDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer)
        self.transforms = A.Compose([
            A.Resize(height=Config.img_size[0], width=Config.img_size[1]),
            ToTensorV2(),
        ])

    def __len__(self) -> int:
        return self.df.shape[0]
    
    def __getitem__(self, ix: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[ix]

        # Image
        image_path = os.path.join('../memotion_dataset_7k/images', row['image_name'].lower())
        img = np.array(Image.open(image_path).convert('RGB'))
        img = self.transforms(image=img)['image']

        # Text
        text = row['text_corrected'].lower()
        out = self.tokenizer(
            text=text, 
            max_length=Config.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        # __import__('pprint').pprint(out)

        return  {
            'image': img, 
            'input_ids': out['input_ids'].squeeze(),
            'attention_mask': out['attention_mask'].squeeze()
        }


if __name__ == '__main__':
    dataset = MemotionDataset(df=pd.read_csv('../memotion_dataset_7k/folds.csv'))
    dataloader = DataLoader(dataset=dataset, batch_size=Config.valid_bs, shuffle=True)
    batch = next(iter(dataloader))
    print(batch.keys())

    __import__('pprint').pprint({
        k: v.shape for k, v in batch.items()
    })

    images = batch['image']
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(5, 8, figsize=(18, 18))
    for ix, ax in enumerate(axs.flatten()):
        ax.set_xticks([])
        ax.set_yticks([])
        if ix > Config.valid_bs:
            break
        img = images[ix].permute(1, 2, 0)
        ax.imshow(img)

    plt.tight_layout()
    plt.show()
