import json
import torch
from numpy import inf
import numpy as np
import torchaudio
from datasets import Dataset, Audio
from typing import Any, Dict, List, Optional, Union
from transformers import AutoFeatureExtractor, AutoTokenizer
from dataclasses import dataclass, field
from pysndfx import AudioEffectsChain

from torch.nn.utils.rnn import pad_sequence

def load_from_json(data_json):
    with open(data_json) as jsonfile:
        x = json.load(jsonfile)
        return x

def save_to_json(data_dict, path):
    with open(path, "w") as write_file:
        json.dump(data_dict, write_file, indent=4)

def cal_class_weight(labels, n_classes, alpha=1.0, epsilon=1e-5, loss_weight_type=2):
    # input: list
    # output: 1-d tensor

    if loss_weight_type == 1:
        labels = np.array(labels)
        class_ratio = np.array([np.sum(labels == (c+1)) for c in range(n_classes)])
        class_ratio = class_ratio / np.sum(class_ratio)
        class_weight = np.power(class_ratio, alpha) / np.sum(
            np.power(class_ratio, alpha)) / (class_ratio + epsilon)
    elif loss_weight_type == 2:
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
    elif loss_weight_type == 3:
        beta=alpha
        labels = np.array(labels)
        n_samples_each = np.array([np.sum(labels == (c+1)) for c in range(n_classes)])
        print(f"class distribution: {n_samples_each}")
        #n_effective = 1.0 - np.power(beta, n_samples_each)
        #class_weight = (1.0 - beta) / np.array(n_effective)
        class_weight = [ ( 1 - beta ) / ( 1 - beta ** n) for n in n_samples_each]
        class_weight = class_weight / np.sum(class_weight) * n_classes

    return torch.Tensor(class_weight)

def speech_file_to_array_fn(path):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def make_dataset(data_json, model_args, do_augment=False):

    print("Loading data from {} ...".format(data_json))
    data_dict = load_from_json(data_json)

    dataset = Dataset.from_dict(data_dict)
    
    # batch[audio] include path, array
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    if "speed_perturb" in model_args and do_augment :
        # NOTE: 3 times of data

        augmented_dataset = copy.deepcopy(dataset)
        for speed_factor in model_args["speed_perturb"]:
            if speed_factor != 1:

                # NOTE: data augmentation
                def speed_pertubation(batch):

                    # NOTE: speed perturbation
                    AE = AudioEffectsChain()
                    AE = AE.speed(speed_factor)
                    fx = (AE)
                    batch["audio"]["array"] = fx(batch["audio"]["array"])
                    # rename id
                    batch["id"] = batch["id"] + "_sp{}".format(speed_factor)

                    return batch

                print("[INFO] speed perturbation for speed {} ...".format(speed_factor))
                augmented_dataset = concatenate_datasets([
                    augmented_dataset,
                    dataset.map(speed_pertubation, num_proc=4)
                ])

        return augmented_dataset

    else:
        return dataset

@dataclass
class DataCollatorWithPadding:
    feature_extractor: AutoFeatureExtractor
    tokenizer: AutoTokenizer
    task_type: str
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

        d_type = torch.long if self.problem_type == "single_label_classification" else torch.float

        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            return_attention_mask=True,
        )
        batch["labels"] = torch.tensor(label_features, dtype=d_type)

        # response (text)
        text_list = [ feature["input_ids"] for feature in features]
        
        text_batch = self.tokenizer(text_list, 
                                max_length=256,
                                padding='max_length', 
                                truncation=True,
                                return_tensors='pt') # ['input_ids', 'token_type_ids', 'attention_mask']

        batch["input_ids"] = text_batch["input_ids"]
        batch["text_attention_mask"] = text_batch["attention_mask"]

        # prompt
        prompt_list = [ feature["prompt_input_ids"] for feature in features]
        
        prompt_batch = self.tokenizer(prompt_list, 
                                max_length=256,
                                padding='max_length', 
                                truncation=True,
                                return_tensors='pt') # ['input_ids', 'token_type_ids', 'attention_mask']

        batch["prompt_input_ids"] = prompt_batch["input_ids"]
        batch["prompt_attention_mask"] = prompt_batch["attention_mask"]
        
        if "delivery" in features[0]:
            # delivery (word-level sequence)
            try:
                delivery = [torch.tensor(feature["delivery"]) for feature in features]
                delivery = pad_sequence(delivery, batch_first=True, padding_value=-1)
                delivery_mask = ~(delivery == -1).all(dim=2)
                delivery_mask = delivery_mask.long()
                batch["delivery"] = delivery
                batch["delivery_mask"] = delivery_mask
            except:
                batch["delivery"] = None
                batch["delivery_mask"] = None
            
        if "language_use" in features[0]:
            # language use (token-level sequence)
            try:
                language_use = [torch.tensor(feature["language_use"]) for feature in features]
                language_use = pad_sequence(language_use, batch_first=True, padding_value=-1)
                language_use_mask = ~(language_use == -1).all(dim=2)
                language_use_mask = language_use_mask.long()
                batch["language_use"] = language_use
                batch["language_use_mask"] = language_use_mask
            except:
                batch["language_use"] = None
                batch["language_use_mask"] = None

        return batch
