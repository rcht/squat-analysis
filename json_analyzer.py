import json
from typing import List, Dict, Set
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
class PureConvClassifier(nn.Module):
    def __init__(self, in_ch=9, conv_ch=[64,128,256], kernel=3, num_classes=6):
        super().__init__()
        layers = []
        prev = in_ch
        for ch in conv_ch:
            layers += [
                nn.Conv1d(prev, ch, kernel, padding=kernel//2),
                nn.BatchNorm1d(ch),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ]
            prev = ch
        self.net = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(prev, prev//2)
        self.fc2 = nn.Linear(prev//2, num_classes)
        
    def forward(self, x_packed):
        x, lengths = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)
        x = x.transpose(1,2)              
        x = self.net(x)                   
        x = self.global_pool(x).squeeze(2) 
        x = F.relu(self.fc1(x))
        return self.fc2(x)

import torch
from torch.utils.data import Dataset

class SquatRepDataset(Dataset):
    def __init__(self, data_list, label_list):
        self.data_list = data_list  # list of dicts like the one you provided
        self.label_list = label_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        sequence = torch.tensor([
            data["knee_angle"],
            data["torso_angle"],
            data["hip_angle"],
            data["symmetry_score"],
            data["alignment_score"],
            data["head_angle"],
            data["heel_angle"],
            data["back_angle"],
            data["inter_thigh_angle"],
        ], dtype=torch.float).T  # shape: [seq_len, num_features]

        label = torch.tensor(self.label_list[idx], dtype=torch.long)
        return sequence, label

def squat_collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([seq.shape[0] for seq in sequences])  # original lengths
    padded_sequences = pad_sequence(sequences, batch_first=True)
    labels = torch.stack(labels)
    return padded_sequences, lengths, labels

def load_json(filepath: str) -> List[Dict]:
    """
    Load rep-wise joint metrics from a JSON file.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        List[Dict]: A list of dictionaries, each representing a rep.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def compute_avg(rep: Dict, key: str) -> float:
    """
    Compute the average of a given metric for a rep.

    Args:
        rep (Dict): Dictionary of rep data.
        key (str): Metric key to average.

    Returns:
        float: Average of the metric.
    """
    return sum(rep[key]) / len(rep[key]) if rep[key] else 0.0


def classify_rep(rep: Dict) -> Set[str]:
    """
    Classify a rep based on averaged metrics.

    Args:
        rep (Dict): A rep dictionary with metrics.

    Returns:
        Set[str]: A set of labels assigned to the rep.
    """
    labels = set()

    avg_knee = compute_avg(rep, "knee_angle")
    avg_torso = compute_avg(rep, "torso_angle")
    avg_symmetry = compute_avg(rep, "symmetry_score")
    avg_alignment = compute_avg(rep, "alignment_score")
    avg_head = compute_avg(rep, "head_angle")
    avg_toe_dist = max(rep['toe_distance'])
    back_angles =min(rep['back_angle'])
    max_back_angle = max(rep['back_angle'])
    min_knee = min(rep['knee_angle'])
    min_heel= min(rep['heel_angle'])
    if avg_knee < 100 and avg_torso < 50 and avg_symmetry < 30 and avg_alignment > 0.8:
        labels.add("good")
    else:
        if min_knee > 60:
            labels.add("bad_shallow")
        if back_angles < 30 or max_back_angle < 100:
            labels.add("bad_back_warp")
        # if min_heel < 75:
        #     labels.add("bad_toe")
        if avg_head > 110 or avg_head < 90:
            labels.add("bad_head")
        if avg_toe_dist > 0.02:
            labels.add("bad_inner_thigh")
    if len(labels)==0:
        labels.add("good")
    return labels


def analyze_json(filepath: str) -> List[Dict]:
    """
    Analyze and classify all reps in a JSON file.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        List[Dict]: List of classification results per rep.
    """
    reps = load_json(filepath)
    results = []

    for rep in reps:
        classification = classify_rep(rep)
        results.append({
            "rep_number": rep["rep_number"],
            "labels": list(classification)
        })

    return results

import torch
import torch.nn as nn
import glob
import os
def analyze_json_lstm(filepath: str, model_path: str) -> List[Dict]:
    """
    Analyze and classify all reps in a JSON file using LSTM model.

    Args:
        filepath (str): Path to the JSON file.
        model_path (str): Path to the LSTM model.

    Returns:
        List[Dict]: List of classification results per rep.
    """
    # Load the LSTM model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PureConvClassifier().to(device)  # Adjust input size and hidden size as needed
    # Load the trained model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()
    file = load_json(filepath)
    actual_list = []
    for j in file:
        actual_list.append(j)
    future=['temp_data/bad_inner_thigh', 'temp_data/bad_shallow', 'temp_data/good', 'temp_data/bad_head', 
            'temp_data/bad_back_warp', 'temp_data/bad_toe']
    inputs = [torch.tensor([
                data["knee_angle"],
                data["torso_angle"],
                data["hip_angle"],
                data["symmetry_score"],
                data["alignment_score"],
                data["head_angle"],
                data["inter_thigh_angle"],
                data["heel_angle"],
                data["back_angle"]
            ], dtype=torch.float).T  
            for data in actual_list]
    results = []
    classes=os.listdir("temp_data")
    mapping = {i: future[i].split('/')[-1] for i in range(len(classes))}
    i=1
    model.eval()
    with torch.no_grad():
        for seq in inputs:
            seq = seq.to(device)

            seq = seq.unsqueeze(0)

            lengths = [seq.size(1)]    

            packed = nn.utils.rnn.pack_padded_sequence(
                seq,
                lengths,
                batch_first=True,
                enforce_sorted=False
            )

            logits = model(packed)     
            probs  = torch.sigmoid(logits)
            preds  = (probs >= 0.5).long()
            # print(torch.argmax(probs))
            # print(f"preds: {preds.squeeze(0)}")
            index=[i for i in range(len(preds.squeeze(0))) if preds.squeeze(0)[i] == 1]
            results.append({
                "rep_number": i,
                "labels": [mapping[i] for i in index]  # Assuming labels are integers
            })
            i+=1
    return results