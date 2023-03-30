from typing import Optional

import torch
import torch.nn as nn
from config import Config
from transformers import AutoModel


class ImageEncoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = AutoModel.from_pretrained(Config.image_encoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder.forward(x)["pooler_output"]
        return x


class TextEncoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = AutoModel.from_pretrained(Config.tokenizer)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        x = self.encoder.forward(input_ids=input_ids, attention_mask=attention_mask)
        return x["pooler_output"]


class MemotionModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.alpha_img = torch.randn(size=(1,), requires_grad=True)
        self.alpha_txt = torch.randn(size=(1,), requires_grad=True)
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 3)
        self.dropout = nn.Dropout(p=0.2)

    def forward(
        self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        img_out = self.image_encoder.forward(image)
        txt_out = self.text_encoder.forward(
            input_ids=input_ids, attention_mask=attention_mask
        )
        wt_emb = self.alpha_txt * txt_out + self.alpha_img * img_out
        x = self.fc1(self.dropout(wt_emb))
        x = self.fc2(self.dropout(x))
        return self.fc3(x)


if __name__ == "__main__":
    import pandas as pd
    from dataset import MemotionDataset
    from torch.utils.data import DataLoader

    dataset = MemotionDataset(df=pd.read_csv("../memotion_dataset_7k/folds.csv"))
    dataloader = DataLoader(dataset=dataset, batch_size=Config.train_bs, shuffle=True)
    batch = next(iter(dataloader))
    img = batch["image"]

    model = MemotionModel()
    y = model.forward(**batch)
    print(y.shape)
