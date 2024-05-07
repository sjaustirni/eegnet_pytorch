import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset


def _get_participant_epochs_file(data_dir, participant_id):
    return os.path.join(data_dir, f"participant{participant_id}_epochs.npy")


class DearEarDataset(Dataset):
    def __init__(self, data_dir: str, participant_ids: [int], mode="train", classes: list = None, scenario: str = "REHAB",
                 channels: list = None):
        """
        Args:
            data_dir:
            participant_id:
            classes:
            scenario:
            channels:
        """
        self.classes = classes
        self.scenario = scenario
        self.channels = channels
        
        if classes is None:
            self.classes = ["hand", "idle"]
        if scenario is None:
            self.scenario = "REHAB"
        if channels is None:
            self.channels = [0, 1, 2]
        
        self.data = []
        self.labels = []
    
        for participant_id in participant_ids:
            data, labels = self.load_data(_get_participant_epochs_file(data_dir, participant_id))
            
            # For training use all data
            if mode == "train":
                self.data.append(data)
                self.labels.append(labels)
                
            # For fine-tuning use only first 50% of each class
            elif mode == "finetune":
                for c in self.classes:
                    class_data = data[labels == self.classes.index(c)]
                    class_labels = labels[labels == self.classes.index(c)]
                    self.data.append(class_data[:len(class_data) // 2])
                    self.labels.append(class_labels[:len(class_labels) // 2])
            
            # For evaluation use only second 50% of each class
            elif mode == "eval":
                for c in self.classes:
                    class_data = data[labels == self.classes.index(c)]
                    class_labels = labels[labels == self.classes.index(c)]
                    self.data.append(class_data[len(class_data) // 2:])
                    self.labels.append(class_labels[len(class_labels) // 2:])
            
            else:
                raise ValueError("Invalid mode")
            
        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
    
    def load_data(self, file):
        # Load data
        data = np.load(file, allow_pickle=True)
        epochs = []
        for c in self.classes:
            class_epochs = data.item()['MRCP'][c][self.scenario][:, self.channels, :]
            epochs.append(class_epochs)
            
        return np.concatenate(epochs, axis=0), np.concatenate(
            [np.ones(len(data.item()['MRCP'][c][self.scenario])) * i for i, c in enumerate(self.classes)])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor):
        return torch.from_numpy(self.data[idx]).float(), torch.tensor(self.labels[idx]).long()
        
