import numpy as np

import torch
from torch.utils.data import Dataset

from datasets import load_dataset
from transformers import BertTokenizer

from textattack.augmentation import EasyDataAugmenter

np.random.seed(42)


class RottenTomatoesDataset(Dataset):
    def __init__(self, split: str, augment: bool = False, p: float = 0.5):
        self.split = split
        self.dataset = load_dataset('rotten_tomatoes')[split]
        self.augment = augment
        self.p = p

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.dataset)

    def augment_text(self, text: str) -> str:
        eda_aug = EasyDataAugmenter()
        new_text = eda_aug.augment(text)
        new_text = new_text[np.random.randint(0,len(new_text))]
        return new_text

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.dataset[idx]
        text, label = sample['text'], sample['label']

        # print(f'Original text: {text}')

        if self.augment:
            # Make an augmentation to one sample with probability p
            if np.random.random() < self.p:
                text = self.augment_text(text)
                # print(f'New text:      {text}')

        encoded_input = self.tokenizer(
            text, 
            max_length=128, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        # Remove extra dimensions
        encoded_input['input_ids'] = encoded_input['input_ids'].squeeze()
        encoded_input['token_type_ids'] = encoded_input['token_type_ids'].squeeze()
        encoded_input['attention_mask'] = encoded_input['attention_mask'].squeeze()
        encoded_input['labels'] = torch.tensor(label)

        return encoded_input
