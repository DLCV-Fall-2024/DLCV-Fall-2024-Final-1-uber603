from torch.utils.data import Dataset, Subset
from PIL import Image
import numpy as np

class CodaDataset(Dataset):
    def __init__(self, hf_dataset, has_answer):
        self.hf_dataset = hf_dataset
        self.has_answer = has_answer

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        data = self.hf_dataset[idx]
        data_id, image, conv = data['id'], data['image'], data['conversations']

        width, height = image.size
        max_dim = max(width, height)
        image = image.resize((max_dim, max_dim))

        _, question_type, _ = data_id.split('_')
        assert question_type in ['general', 'regional', 'suggestion']

        if self.has_answer:
            assert len(conv) == 2
            assert conv[0]['from'] == 'human'
            assert conv[1]['from'] == 'gpt'
            question = conv[0]['value']
            answer = conv[1]['value']
            return data_id, question_type, image, question, answer
        else:
            assert len(conv) == 1
            assert conv[0]['from'] == 'human'
            question = conv[0]['value']
            return data_id, question_type, image, question

class DummyDataset(Dataset):
    def __init__(self, length, has_answer):
        self.length = length
        self.has_answer = has_answer

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx >= len(self): raise IndexError

        if idx % 2 == 0:
            data_id = 'dummy_data_id_0'
            question_type = 'dummy_question_type_0'
            image = Image.new('RGB', (100, 100), (255, 0, 0))
            question = '<image>\n What is the color of the image?'
            answer = 'The quick brown fox jumps over the lazy dog.'
        else:
            data_id = 'dummy_data_id_1'
            question_type = 'dummy_question_type_1'
            noise = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            image = Image.fromarray(noise, mode='RGB')
            question = '<image>\n What is the color of the image?'
            answer = 'DLCV 2024-Fall Final Competition Topic 1'

        if self.has_answer:
            return data_id, question_type, image, question, answer
        else:
            return data_id, question_type, image, question

class CodaAugmentedDataset(Dataset):
    def __init__(self, hf_dataset, qa_pairs):
        self.hf_dataset = hf_dataset
        self.qa_pairs = qa_pairs

        data_ids = self.hf_dataset['id']
        self.id_to_index = {data_id: idx for idx, data_id in enumerate(data_ids)}

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        data_id, question, answer = self.qa_pairs[idx]
        image = self.hf_dataset[self.id_to_index[data_id]]['image']

        width, height = image.size
        max_dim = max(width, height)
        image = image.resize((max_dim, max_dim))

        question = f'<image>\n{question}'

        return data_id, None, image, question, answer

class CodaCombinedDataset(Dataset):
    def __init__(self, original_dataset, augmented_dataset):
        self.original_dataset = original_dataset
        self.augmented_dataset = augmented_dataset

    def __len__(self):
        return len(self.original_dataset) + len(self.augmented_dataset)

    def __getitem__(self, idx):
        if idx >= len(self.original_dataset):
            idx -= len(self.original_dataset)
            return self.augmented_dataset[idx]
        else:
            return self.original_dataset[idx]

def get_coda_subset_dataset(original_dataset, id_to_index, qa_pairs):
    subset_indices = list(set(id_to_index[data_id] for data_id, _, _ in qa_pairs))
    return Subset(original_dataset, subset_indices)