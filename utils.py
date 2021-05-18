from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_classifier_head import ClassificationHead


DISCRIMINATOR_MODELS_PARAMS = {
    "clickbait": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/clickbait_classifier_head.pt",
        "class_size": 2,
        "embed_size": 1024,
        "class_vocab": {"non_clickbait": 0, "clickbait": 1},
        "default_class": 1,
        "pretrained_model": "gpt2-medium",
    },
    # "sentiment": {
    #     "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/SST_classifier_head.pt",
    #     "class_size": 5,
    #     "embed_size": 1024,
    #     "class_vocab": {"very_positive": 2, "very_negative": 3},
    #     "default_class": 3,
    #     "pretrained_model": "gpt2-medium",
    # },
    "sentiment": {
        "path": "/home/dianyu/trigger/sst_head/SST_classifier_head_epoch_10.pt",
        "class_size": 5,
        "embed_size": 1024,
        "class_vocab": {"very_positive": 2, "very_negative": 3},
        "default_class": 3,
        "pretrained_model": "gpt2-medium",
    },

    "safety": {
        "path": "semi_safety_cls/semi_safety_cls_classifier_head_epoch_3.pt",
        "class_size": 2,
        "embed_size": 1280,
        "class_vocab": {"__notok__": 0, "__ok__": 1},
        "default_class": 0,
    },

    "contra": {
        "path": "semi_decode_cls/semi_decode_cls_classifier_head_epoch_3.pt",
        "class_size": 2,
        "embed_size": 1280,
        "class_vocab": {"is_contradiction": 0, "not_contradiction": 1},
        "default_class": 0,
    }


}


def get_classifier(
    name: Optional[str], class_label: Union[str, int], device: str
) -> Tuple[Optional[ClassificationHead], Optional[int]]:
    if name is None:
        return None, None

    params = DISCRIMINATOR_MODELS_PARAMS[name]
    classifier = ClassificationHead(class_size=params["class_size"], embed_size=params["embed_size"]).to(device)
    if "url" in params:
        resolved_archive_file = cached_path(params["url"])
    elif "path" in params:
        resolved_archive_file = params["path"]
    else:
        raise ValueError("Either url or path have to be specified in the discriminator model parameters")
    classifier.load_state_dict(torch.load(resolved_archive_file, map_location=device))
    classifier.eval()

    try:
        class_label = int(class_label)  # class_label is str from input
    except:
        pass

    if isinstance(class_label, str):
        if class_label in params["class_vocab"]:
            label_id = params["class_vocab"][class_label]
        else:
            label_id = params["default_class"]
            print("class_label {} not in class_vocab".format(class_label))
            print("available values are: {}".format(params["class_vocab"]))
            print("using default class {}".format(label_id))

    elif isinstance(class_label, int):
        if class_label in set(params["class_vocab"].values()):
            label_id = class_label
        else:
            label_id = params["default_class"]
            print("class_label {} not in class_vocab".format(class_label))
            print("available values are: {}".format(params["class_vocab"]))
            print("using default class {}".format(label_id))

    else:
        label_id = params["default_class"]

    return classifier, label_id


# Gets the score for the top-k logits to improve quality of samples.
def top_k_logits(logits, k):
    if k == 0:
        return logits
    values = torch.topk(logits, k)[0]
    batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
    return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)


def generate_next(logits, output_so_far, top_k=10, temperature=1.0, repetition_penalty=1.0, sample=True,
                  gumbel_softmax=False, gumbel_temperature=1.0, detach=False, ):
    logits = logits[:, -1, :] / temperature
    logits = top_k_logits(logits, top_k)

    if repetition_penalty > 1:
        for i, output_so_far_i in enumerate(output_so_far):
            for token_idx in set(output_so_far_i.tolist()):
                if logits[i, token_idx] < 0:
                    logits[i, token_idx] *= repetition_penalty
                else:
                    logits[i, token_idx] /= repetition_penalty

    probs = F.softmax(logits, dim=-1)

    if sample:
        if gumbel_softmax:  # note: it's a one-hot vector now
            gumbel_vector = F.gumbel_softmax(logits, tau=gumbel_temperature, hard=True)
            last = torch.argmax(gumbel_vector, dim=-1).unsqueeze(-1)  # shape: bze, 1
            if detach:
                # all_gumbel_vectors.append(gumbel_vector)
                return last, gumbel_vector
        else:
            last = torch.multinomial(probs, num_samples=1)
    else:
        _, last = torch.topk(probs, k=1, dim=-1)

    return last, None


def concat_past(ori_past, new_past, num_layers):
    concated_past = list()
    for layer in range(num_layers):
        l_concat_key = torch.cat((ori_past[layer][0], new_past[layer][0]), dim=-2)
        l_concat_value = torch.cat((ori_past[layer][1], new_past[layer][1]), dim=-2)
        concated_past.append((l_concat_key, l_concat_value))
    return tuple(concated_past)


def expand_past(past, num_layers, batch_size):
    # past: (num_layers, 2, (bze, num_heads, seq_len, embed_per_head))
    new_past = list()
    for layer in range(num_layers):
        key_layer, value_layer = past[layer]
        batch_key_layer = key_layer.repeat(batch_size, 1, 1, 1)
        batch_value_layer = value_layer.repeat(batch_size, 1, 1, 1)
        new_past.append((batch_key_layer, batch_value_layer))
    return tuple(new_past)


def read_file(filename):
    l = list()
    for line in open(filename):
        l.append(line.strip())
    return l
