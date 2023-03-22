import json
import torch
from numpy import inf
import numpy as np
import torchaudio
from datasets import Dataset, Audio
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2FeatureExtractor
from dataclasses import dataclass, field

def load_from_json(data_json):
    with open(data_json) as jsonfile:
        x = json.load(jsonfile)
        return x

def save_to_json(data_dict, path):
    with open(path, "w") as write_file:
        json.dump(data_dict, write_file, indent=4)

def cal_class_weight(labels, n_classes, alpha=1.0, epsilon=1e-5):
    # input: list
    # output: 1-d tensor

    # normal re-weighting
    labels = np.array(labels)
    n_samples = len(labels)
    n_samples_each = np.zeros(n_classes)
    for c in range(n_classes):
        indices = np.where(labels == (c+1))
        n_samples_each[c] = len(labels[indices])
    #class_weight = np.power(n_samples, alpha) / n_classes * np.power(n_samples_each, alpha)
    class_weight = np.power(n_samples, alpha) / np.power(n_samples_each, alpha)
    class_weight[np.isinf(class_weight)] = 0

    '''
    # cefr-sp
    labels = np.array(labels)
    class_ratio = np.array([np.sum(labels == (c+1)) for c in range(n_classes)])
    class_ratio = class_ratio / np.sum(class_ratio)
    class_weight = np.power(class_ratio, alpha) / np.sum(
        np.power(class_ratio, alpha)) / (class_ratio + epsilon)
    '''

    return torch.Tensor(class_weight)

def speech_file_to_array_fn(path):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def make_dataset(data_json):

    print("Loading data from {} ...".format(data_json))
    data_dict = load_from_json(data_json)

    dataset = Dataset.from_dict(data_dict)
    # batch[audio] include path, array
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    return dataset

@dataclass
class DataCollatorCTCWithPadding:
    feature_extractor: Wav2Vec2FeatureExtractor
    problem_type: str
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [feature["labels"] for feature in features]
        #if self.problem_type == "single_label_classification":
        #    label_features = [int(feature["labels"]) for feature in features]
        #    d_type = torch.long
        #else:
        #    label_features = [feature["labels"] for feature in features]
        #    d_type = torch.float
        d_type = torch.long if self.problem_type == "single_label_classification" else torch.float

        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(label_features, dtype=d_type)

        return batch
