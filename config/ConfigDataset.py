from eegnet.data.DearEarDataset import DearEarDataset
import os

class ConfigDataset:
    def __init__(self, evaluation_subject=1, channels: list = [0, 1, 2], classes: list = ["hand", "idle"], scenario: str = "REHAB"):
        all_subjects = [int(el.lstrip("participant").rstrip("_epochs.npy")) for el in os.listdir("data/epochs")]
        
        self.train_subjects = [el for el in all_subjects if el != evaluation_subject]
        self.evaluation_subject = evaluation_subject
        self.channels = channels
        self.classes = classes
        self.scenario = scenario
        
    def make_train_dataset(self) -> DearEarDataset:
        return DearEarDataset("data/epochs", self.train_subjects, mode="train", classes=self.classes, scenario=self.scenario, channels=self.channels)
    
    def make_finetune_dataset(self) -> DearEarDataset:
        return DearEarDataset("data/epochs", [self.evaluation_subject], mode="finetune", classes=self.classes, scenario=self.scenario, channels=self.channels)
    
    def make_eval_dataset(self) -> DearEarDataset:
        return DearEarDataset("data/epochs", [self.evaluation_subject], mode="eval", classes=self.classes, scenario=self.scenario, channels=self.channels)
        