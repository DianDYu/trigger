#! /usr/bin/env python3
# coding=utf-8

# modified from PPLM run_pplm_discrim_train.py
"""python train_blender_classifier_head.py --dataset semi_safety_cls --dataset_fp data/bbf_bad_train.csv --eval_dataset_fp data/bad_valid.csv --save_model --pretrained_model facebook/blenderbot-400M-distill"""

import argparse
import csv
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from nltk.tokenize.treebank import TreebankWordDetokenizer
from torchtext import data as torchtext_data
from torchtext import datasets
from tqdm import tqdm, trange

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import BlenderbotTokenizer, BlenderbotModel, BlenderbotConfig

from utils import expand_past, concat_past

from sklearn.metrics import f1_score


torch.manual_seed(0)
np.random.seed(0)
EPSILON = 1e-10
example_sentence = "This is incredible! I love it, this is the best chicken I have ever had."
max_length_seq = 128

pad_token_id = 0
bos_token_id = 1
eos_token_id = 2

print("max length: %d" % max_length_seq)


class ClassificationHead(torch.nn.Module):
    """Classification Head for  transformer encoders"""

    def __init__(self, class_size, embed_size):
        super().__init__()
        self.class_size = class_size
        self.embed_size = embed_size
        # self.mlp1 = torch.nn.Linear(embed_size, embed_size)
        # self.mlp2 = (torch.nn.Linear(embed_size, class_size))
        self.mlp = torch.nn.Linear(embed_size, class_size)

    def forward(self, hidden_state):
        # hidden_state = F.relu(self.mlp1(hidden_state))
        # hidden_state = self.mlp2(hidden_state)
        logits = self.mlp(hidden_state)
        return logits


class Discriminator(torch.nn.Module):
    """Transformer encoder followed by a Classification Head"""

    def __init__(self, class_size, pretrained_model="facebook/blenderbot-400M-distill", cached_mode=False, device="cpu"):
        super().__init__()
        self.tokenizer = BlenderbotTokenizer.from_pretrained(pretrained_model)
        self.model = BlenderbotModel.from_pretrained(pretrained_model)
        self.embed_size = self.model.config.d_model
        self.classifier_head = ClassificationHead(class_size=class_size, embed_size=self.embed_size)
        self.cached_mode = cached_mode
        self.device = device

        self.model.eval()
        self.model.to(device)
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        self.num_enc_layers = self.model.config.encoder_layers
        # initialize bos
        self.bos_embeddings = self.model.encoder.embed_tokens(
            torch.tensor([bos_token_id], dtype=torch.long, device="cuda")).unsqueeze(0)  # 1, 1, hid_size
        # get_bos_key_values
        text_bos = ["<s>"]
        inputs_bos = self.tokenizer(text_bos, return_tensors='pt', padding=True).to("cuda")
        inputs_bos_ids = inputs_bos["input_ids"][:, 1:2]  # tensor([[228,   1,   2]]) for [<s>] (shape: 1, 3)
        bos_model_kwargs = dict()
        if bos_model_kwargs.get("attention_mask", None) is None:
            # init `attention_mask` depending on `pad_token_id`
            bos_model_kwargs["attention_mask"] = self.model._prepare_attention_mask_for_generation(
                inputs_bos_ids, 0, 2
            )

        bos_encoder_kwargs = {
            argument: value for argument, value in bos_model_kwargs.items() if not argument.startswith("decoder_")
        }
        bos_output = self.model.encoder(inputs_bos_ids, return_dict=True, **bos_encoder_kwargs, use_cache=True)
        self.bos_key_values = bos_output["past_key_values"]
        self.bos_hidden = bos_output["last_hidden_state"]  # 1, 1, 1280

    def get_classifier(self):
        return self.classifier_head

    def train_custom(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        self.classifier_head.train()

    def avg_representation(self, x):
        mask = x.ne(0).unsqueeze(2).repeat(1, 1, self.embed_size).float().to(self.device).detach()
        hidden = self.encoder.transformer(x)["last_hidden_state"]
        masked_hidden = hidden * mask
        avg_hidden = torch.sum(masked_hidden, dim=1) / (torch.sum(mask, dim=1).detach() + EPSILON)
        return avg_hidden

    def get_representation(self, ctx, res):  # ctx, res are tensors (padded)
        # assume 1 trigger, key_value
        batch_size = ctx.shape[0]
        past = expand_past(self.bos_key_values, self.num_enc_layers, batch_size)
        past = concat_past(past, past, self.num_enc_layers)

        # prepare hidden
        prev_hidden = self.bos_hidden.repeat(batch_size, 1, 1)
        prev_hidden = torch.cat((prev_hidden, prev_hidden), dim=1)

        ctx_attn_mask = ctx.ne(0).long().to(self.device)
        # prepare context
        prev_length = prev_hidden.shape[1]
        ctx_model_kwargs = dict()
        # because of the past, now key length ("tgt" as defined in blenderbot) is larger than query length ("tgt" as defined)
        cat_attn_mask = torch.cat((torch.ones(batch_size, prev_length, device=self.device, dtype=torch.long), ctx_attn_mask), dim=-1)
        ctx_model_kwargs["attention_mask"] = cat_attn_mask

        # get encoder output
        trigger_encoder_kwargs = {
                argument: value for argument, value in ctx_model_kwargs.items() if not argument.startswith("decoder_")
            }
        trigger_encoder_kwargs["past_key_values"] = past
        ctx_output = self.model.encoder(ctx, return_dict=True, **trigger_encoder_kwargs, is_trigger=True)
        ctx_output["last_hidden_state"] = torch.cat((prev_hidden, ctx_output["last_hidden_state"]), dim=1)
        ctx_model_kwargs["encoder_outputs"] = ctx_output

        # prepare decoder
        # prompt_inputs = tokenizer(p_texts, return_tensors='pt', padding=True, truncation=True).to("cuda")
        # prompt_inputs_ids = prompt_inputs["input_ids"]
        # prompt_attn_mask = prompt_inputs["attention_mask"]
        res_attn_mask = res.ne(0).long().to(self.device)

        # add bos
        dec_bos_ids = torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * 1
        dec_bos_mask = torch.ones((batch_size, 1), dtype=torch.long, device=self.device)
        dec_inputs_ids = torch.cat((dec_bos_ids, res), dim=1)
        dec_attn_mask = torch.cat((dec_bos_mask, res_attn_mask), dim=1)  # bze x seq

        # prompt_length = torch.sum(dec_attn_mask, dim=-1)  # including bos and eos. shape: [bze]

        model_inputs = {"decoder_input_ids": dec_inputs_ids, "encoder_outputs": ctx_model_kwargs["encoder_outputs"],
                        "attention_mask": ctx_model_kwargs["attention_mask"]}
        outputs = self.model(**model_inputs, return_dict=True, output_hidden_states=True)
        last_hidden = outputs["decoder_hidden_states"][-1]  #  bze x seq_len x hid

        dec_attn_mask = dec_attn_mask.unsqueeze(2).repeat(1, 1, self.embed_size)
        masked_hidden = last_hidden * dec_attn_mask

        avg_hidden = torch.sum(masked_hidden, dim=1) / (torch.sum(dec_attn_mask, dim=1).detach() + EPSILON)
        return avg_hidden
        
    def forward(self, ctx_x, text_x):
        if self.cached_mode:
            avg_hidden = x.to(self.device)
            assert False
        else:
            # avg_hidden = self.avg_representation(x.to(self.device))
            avg_hidden = self.get_representation(ctx_x, text_x)

        logits = self.classifier_head(avg_hidden)
        probs = F.log_softmax(logits, dim=-1)

        return probs


class Dataset(data.Dataset):
    def __init__(self, ctx_X, text_X, y):
        """Reads source and target sequences from txt files."""
        self.ctx_X = ctx_X
        self.text_X = text_X
        self.y = y

    def __len__(self):
        return len(self.ctx_X)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        data = {}
        data["ctx_X"] = self.ctx_X[index]
        data["text_X"] = self.text_X[index]
        data["y"] = self.y[index]
        return data


def collate_fn(data):
    def pad_sequences(sequences):
        lengths = [len(seq) for seq in sequences]

        padded_sequences = torch.zeros(len(sequences), max(lengths)).long()  # padding value = 0

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_sequences[i, :end] = seq[:end]

        return padded_sequences, lengths

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    ctx_x_batch, _ = pad_sequences(item_info["ctx_X"])
    text_x_batch, _ = pad_sequences(item_info["text_X"])
    y_batch = torch.tensor(item_info["y"], dtype=torch.long)

    return ctx_x_batch, text_x_batch, y_batch


def cached_collate_fn(data):
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    ctx_x_batch = torch.cat(item_info["ctx_X"], 0)
    text_x_batch = torch.cat(item_info["text_X"], 0)
    y_batch = torch.tensor(item_info["y"], dtype=torch.long)

    return ctx_x_batch, text_x_batch, y_batch


def train_epoch(data_loader, discriminator, optimizer, epoch=0, log_interval=10, device="cpu"):
    samples_so_far = 0
    discriminator.train_custom()
    for batch_idx, (ctx_t, text_t, target_t) in enumerate(data_loader):
        ctx_t, text_t, target_t = ctx_t.to(device), text_t.to(device), target_t.to(device)

        optimizer.zero_grad()

        output_t = discriminator(ctx_t, text_t)
        loss = F.nll_loss(output_t, target_t)
        loss.backward(retain_graph=True)
        optimizer.step()

        samples_so_far += len(ctx_t)

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch + 1,
                    samples_so_far,
                    len(data_loader.dataset),
                    100 * samples_so_far / len(data_loader.dataset),
                    loss.item(),
                )
            )


def evaluate_performance(data_loader, discriminator, device="cpu"):
    discriminator.eval()
    test_loss = 0
    correct = 0

    all_pred = None
    all_label = None
    with torch.no_grad():
        for ctx_t, text_t, target_t in data_loader:
            ctx_t, text_t, target_t = ctx_t.to(device), text_t.to(device), target_t.to(device)
            output_t = discriminator(ctx_t, text_t)
            # sum up batch loss
            test_loss += F.nll_loss(output_t, target_t, reduction="sum").item()
            # get the index of the max log-probability
            pred_t = output_t.argmax(dim=1, keepdim=True)
            correct += pred_t.eq(target_t.view_as(pred_t)).sum().item()

            if all_pred is None:
                all_pred = pred_t
                all_label = target_t
            else:
                all_pred = torch.cat((all_pred, pred_t), dim=0)
                all_label = torch.cat((all_label, target_t), dim=0)

    f1 = f1_score(y_true=all_label.cpu().numpy(), y_pred=all_pred.cpu().numpy(), pos_label=0)

    test_loss /= len(data_loader.dataset)

    print(
        "Performance on test set: "
        "Average loss: {:.4f}, F1: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss, f1, correct, len(data_loader.dataset), 100.0 * correct / len(data_loader.dataset)
        )
    )


def predict(input_sentence, model, classes, cached=False, device="cpu"):
    input_t = model.tokenizer.encode(input_sentence)
    input_t = torch.tensor([input_t], dtype=torch.long, device=device)
    if cached:
        input_t = model.avg_representation(input_t)

    log_probs = model(input_t).data.cpu().numpy().flatten().tolist()
    print("Input sentence:", input_sentence)
    print(
        "Predictions:",
        ", ".join("{}: {:.4f}".format(c, math.exp(log_prob)) for c, log_prob in zip(classes, log_probs)),
    )


def get_cached_data_loader(dataset, batch_size, discriminator, shuffle=False, device="cpu"):
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_fn)

    xs = []
    ys = []
    for batch_idx, (x, y) in enumerate(tqdm(data_loader, ascii=True)):
        with torch.no_grad():
            x = x.to(device)
            avg_rep = discriminator.avg_representation(x).cpu().detach()
            avg_rep_list = torch.unbind(avg_rep.unsqueeze(1))
            xs += avg_rep_list
            ys += y.cpu().numpy().tolist()

    data_loader = torch.utils.data.DataLoader(
        dataset=Dataset(xs, ys), batch_size=batch_size, shuffle=shuffle, collate_fn=cached_collate_fn
    )

    return data_loader


def get_data(discriminator, datapath, class2idx, device):
    ctx_x = []
    text_x = []
    y = []
    with open(datapath) as f:
        csv_reader = csv.DictReader(f)
        for i, row in enumerate(tqdm(csv_reader, ascii=True)):
            if row:
                label = row["label"]
                text = row["text"]
                context = row["context"]

                try:
                    text_seq = discriminator.tokenizer(text, truncation=True)["input_ids"]
                    ctx_seq = discriminator.tokenizer(context, truncation=True)["input_ids"]
                    if len(ctx_seq) < max_length_seq and len(text_seq) < max_length_seq:
                        text_seq = torch.tensor(text_seq, device=device, dtype=torch.long)
                        ctx_seq = torch.tensor(ctx_seq, device=device, dtype=torch.long)

                    else:
                        # print("Line {} is longer than maximum length {}".format(i, max_length_seq))
                        continue

                    ctx_x.append(ctx_seq)
                    text_x.append(text_seq)
                    y.append(class2idx[label])

                except Exception as e:
                    print(e)
                    assert False
                    print("Error tokenizing line {}, skipping it".format(i))
                    pass

    full_dataset = Dataset(ctx_x, text_x, y)
    return full_dataset


def train_discriminator(
    dataset,
    dataset_fp=None,
    eval_dataset_fp=None,
    pretrained_model="gpt2-medium",
    epochs=10,
    batch_size=64,
    log_interval=10,
    save_model=False,
    cached=False,
    no_cuda=False,
):
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"

    print("Preprocessing {} dataset...".format(dataset))
    start = time.time()

    if dataset == "test":
        assert False
    else:  # if dataset == "generic":
        # This assumes the input dataset is a TSV with the following structure:
        # class \t text

        if dataset_fp is None:
            raise ValueError("When generic dataset is selected, " "dataset_fp needs to be specified aswell.")

        classes = set()
        with open(dataset_fp) as f:
            csv_reader = csv.DictReader(f)
            for row in tqdm(csv_reader, ascii=True):
                if row:
                    # print(row)
                    classes.add(row['label'])

        idx2class = sorted(classes)
        class2idx = {c: i for i, c in enumerate(idx2class)}

        discriminator = Discriminator(
            class_size=len(idx2class), pretrained_model=pretrained_model, cached_mode=cached, device=device
        ).to(device)

        ctx_x = []
        text_x = []
        y = []
        with open(dataset_fp) as f:
            csv_reader = csv.DictReader(f)
            for i, row in enumerate(tqdm(csv_reader, ascii=True)):
                if row:
                    label = row["label"]
                    text = row["text"]
                    context = row["context"]

                    try:
                        text_seq = discriminator.tokenizer(text, truncation=True)["input_ids"]
                        ctx_seq = discriminator.tokenizer(context, truncation=True)["input_ids"]
                        if len(ctx_seq) < max_length_seq and len(text_seq) < max_length_seq:
                            text_seq = torch.tensor(text_seq, device=device, dtype=torch.long)
                            ctx_seq = torch.tensor(ctx_seq, device=device, dtype=torch.long)

                        else:
                            # print("Line {} is longer than maximum length {}".format(i, max_length_seq))
                            continue

                        ctx_x.append(ctx_seq)
                        text_x.append(text_seq)
                        y.append(class2idx[label])

                    except Exception as e:
                        print(e)
                        assert False
                        print("Error tokenizing line {}, skipping it".format(i))
                        pass

        if eval_dataset_fp is not None:
            train_dataset = get_data(discriminator, dataset_fp, class2idx, device)
            test_dataset = get_data(discriminator, eval_dataset_fp, class2idx, device)
            train_size = len(train_dataset)
            test_size = len(test_dataset)
        else:
            full_dataset = get_data(discriminator, dataset_fp, class2idx, device)
            train_size = int(0.9 * len(full_dataset))
            test_size = len(full_dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

        discriminator_meta = {
            "class_size": len(idx2class),
            "embed_size": discriminator.embed_size,
            "pretrained_model": pretrained_model,
            "class_vocab": class2idx,
            "default_class": 0,
        }

    end = time.time()
    print("Preprocessed {} data points".format(len(train_dataset) + len(test_dataset)))
    print("Data preprocessing took: {:.3f}s".format(end - start))

    if cached:
        print("Building representation cache...")

        start = time.time()

        train_loader = get_cached_data_loader(train_dataset, batch_size, discriminator, shuffle=True, device=device)

        test_loader = get_cached_data_loader(test_dataset, batch_size, discriminator, device=device)

        end = time.time()
        print("Building representation cache took: {:.3f}s".format(end - start))

    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    if save_model:
        with open("{}_classifier_head_meta.json".format(dataset), "w") as meta_file:
            json.dump(discriminator_meta, meta_file)

    optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)

    for epoch in range(epochs):
        start = time.time()
        print("\nEpoch", epoch + 1)

        train_epoch(
            discriminator=discriminator,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            log_interval=log_interval,
            device=device,
        )
        evaluate_performance(data_loader=test_loader, discriminator=discriminator, device=device)

        end = time.time()
        print("Epoch took: {:.3f}s".format(end - start))

        # print("\nExample prediction")
        # predict(example_sentence, discriminator, idx2class, cached=cached, device=device)

        if save_model:
            # torch.save(discriminator.state_dict(),
            #           "{}_discriminator_{}.pt".format(
            #               args.dataset, epoch + 1
            #               ))
            torch.save(
                discriminator.get_classifier().state_dict(),
                "{}_classifier_head_epoch_{}.pt".format(dataset, epoch + 1),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a discriminator on top of model representations")
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset to train the discriminator on."
        "In case of generic, the dataset is expected"
        "to be a TSBV file with structure: class \\t text",
    )
    parser.add_argument(
        "--dataset_fp",
        type=str,
        default="",
        help="File path of the dataset to use. "
    )
    parser.add_argument(
        "--eval_dataset_fp",
        type=str,
        default=None,
        help="File path of the dataset to use for testing. "
    )
    parser.add_argument(
        "--pretrained_model", type=str, help="Pretrained model to use as encoder"
    )
    parser.add_argument("--epochs", type=int, default=10, metavar="N", help="Number of training epochs")
    parser.add_argument(
        "--batch_size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--save_model", action="store_true", help="whether to save the model")
    parser.add_argument("--cached", action="store_true", help="whether to cache the input representations")
    parser.add_argument("--no_cuda", action="store_true", help="use to turn off cuda")
    args = parser.parse_args()

    train_discriminator(**(vars(args)))
