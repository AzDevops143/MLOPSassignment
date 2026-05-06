import torch
from sklearn.metrics import accuracy_score

# Label mappings derived from the notebook state
label2id = {'history_biography': 0, 'children': 1, 'mystery_thriller_crime': 2, 'young_adult': 3, 'fantasy_paranormal': 4, 'comics_graphic': 5, 'romance': 6, 'poetry': 7}
id2label = {v: k for k, v in label2id.items()}

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}
