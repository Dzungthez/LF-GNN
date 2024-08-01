from sklearn.model_selection import train_test_split
import json
import os
import pandas as pd
from transformers import LongformerTokenizer
import torch
from torch.utils.data import Dataset
list_skipped_words = ['should', 'did', 'must', 'just', '.', '..','...', 
                      'the', 'a', 'an', 'in', 'on', 'at', 'to', 'of', 'for', 
                      'with', 'by', 'and', 'or', 'but', 'so', 'nor', 'yet', 
                      'from', 'into', 'onto', 'upon', 'out', 'off', 'over', 
                      'under', 'below', 'above', 'between', 'among', 'through', 
                      'during', 'before', 'after', 'since', 'until', 'while', 
                      'as', 'like', 'about', 'against', 'among', 'around']


def load_fulltext_raw_dir(input_path):
    """
    Load full text of case law from raw data folder.

    Each file contains two key 'paragraphs' and 'meta', containing a list of all paragraphs.
    Our aim is to merge all paragraphs to load the full text of case law.

    return: dictionary of case law with key as case_id, value as full text of case law.
    """

    case_law = {}
    for filename in os.listdir(input_path):
        if filename.endswith(".json"):
            file_path = os.path.join(input_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    temp_data = json.load(f)
                    case_id = filename.split('.')[0]
                    fulltext = temp_data.get('meta', '') + ' '
                    for par in temp_data.get('paragraphs', []):
                        fulltext += par + ' '
                    case_law[case_id] = fulltext.strip()
            except FileNotFoundError:
                print(f"File not found: {file_path}")
            except json.JSONDecodeError:
                print(f"Error decoding JSON from file: {file_path}")
            except Exception as e:
                print(f"An error occurred while processing file {file_path}: {e}")
    return case_law

def filter_useless_words(fulltext):
    """
    remove useless words, give a str that contains full text of a single case law

    return: result string
    """
    words = fulltext.split()
    filtered_words = [word for word in words if word.lower() not in list_skipped_words]
    return ' '.join(filtered_words)


def create_train_val_dataframe(input_path, labels_path, filter = False):
    """
    Split case law into training and validation set
    input_path: path to the input_data_folder
    return: train_df, val_df
    """
    case_law = load_fulltext_raw_dir(input_path)
    if filter:
        for key, value in case_law.items():
            case_law[key] = filter_useless_words(value)
    examples = []
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    for key, values in labels.items():
        case_id = key
        for nested_case_id, _label in values.items():
            if case_id in case_law:
                examples.append({
                    'text1': case_law[case_id],
                    'text2': case_law[nested_case_id],
                    'text1_id' : case_id,
                    'text2_id' : nested_case_id,
                    'label': _label})

    train_examples, val_examples = train_test_split(examples, test_size=0.2, random_state=42)
    train_df = pd.DataFrame(train_examples)
    val_df = pd.DataFrame(val_examples)

    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    return train_df, val_df


class ColieeDataset(Dataset):
    def __init__(self, data_df, max_len, lawyer_idx_to_embs, lawyer_to_idx, case_lawyer_mapping, device):
        try:
            self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
            self.max_len = max_len
            self.data = data_df
            self.lawyer_idx_to_embs = lawyer_idx_to_embs
            self.lawyer_to_idx = lawyer_to_idx
            self.case_lawyer_mapping = case_lawyer_mapping
            self.device = device
        except Exception as e:
            print(f"Error initializing the dataset: {e}")

    def __len__(self):
        try:
            return len(self.data)
        except Exception as e:
            print(f"Error getting dataset length: {e}")
            return 0

    def __getitem__(self, idx):
        try:
            item = dict(self.data.iloc[idx])
            item = self.tokenize_text_pair(item)
            return item
        except Exception as e:
            print(f"Error getting item at index {idx}: {e}")
            return None

    def tokenize_text_pair(self, item):
        try:
            inputs = self.tokenizer(
                item['text1'], item['text2'], 
                padding='max_length', 
                truncation='longest_first', 
                max_length=self.max_len, 
                return_tensors='pt'
            )
            inputs = {key: val.squeeze(0).to(self.device) for key, val in inputs.items()}
            inputs['labels'] = torch.tensor(item['label']).to(self.device)
            
            node1 = self.get_node_embeddings(item['text1_id'])
            node2 = self.get_node_embeddings(item['text2_id'])
            inputs['node1'] = node1
            inputs['node2'] = node2
            return inputs
        except Exception as e:
            print(f"Error tokenizing text pair: {e}")
            return {}

    def get_node_embeddings(self, case_id: str):
        try:
            if case_id not in self.case_lawyer_mapping:
                return torch.zeros((128, 1)).to(self.device)
            lawyer_ids = self.case_lawyer_mapping[case_id]
            lawyer_embeddings = [self.lawyer_idx_to_embs[self.lawyer_to_idx[lawyer]].unsqueeze(1) for lawyer in lawyer_ids]
            if not lawyer_embeddings:
                return torch.zeros((128, 1)).to(self.device) 
            ans = torch.cat(lawyer_embeddings, dim=1)
            return ans.to(self.device)
        except Exception as e:
            return torch.zeros((128, 1)).to(self.device)


