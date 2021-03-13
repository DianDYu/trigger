import argparse
import logging
import random
import time
import os
from typing import List, Optional, Tuple, Union
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from transformers import AdamW, get_linear_schedule_with_warmup

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


def concat_past(ori_past, new_past, num_layers):
    concated_past = list()
    for layer in range(num_layers):
        l_concat_key = torch.cat((ori_past[layer][0], new_past[layer][0]), dim=-2)
        l_concat_value = torch.cat((ori_past[layer][1], new_past[layer][1]), dim=-2)
        concated_past.append((l_concat_key, l_concat_value))
    return tuple(concated_past)


# Gets the score for the top-k logits to improve quality of samples.
def top_k_logits(logits, k):
    if k == 0:
        return logits
    values = torch.topk(logits, k)[0]
    batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
    return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)


def generate_prompt(
    model,
    tokenizer,
    num_of_triggers,
    learning_rate=5e-5,
    adam_epsilon=1e-8,
    context=None,
    device="cuda",
    discrim=None,
    class_label=None,
    trigger_format="token",
    not_mask_trigger=False,
    length=100,
    temperature=1.0,
    top_k=10,
    sample=False,
    num_iterations=10,
    repetition_penalty=1.0,
):
    classifier, class_id = get_classifier(discrim, class_label, device)
    
    for param in model.parameters():
        param.requires_grad = False

    num_layers = model.config.n_layer

    if num_of_triggers > 0:
        if trigger_format == "token":  # learn a continuous embedding
            trigger_embedding_list = []
            for _ in range(num_of_triggers):
                trigger_embedding_i = copy.deepcopy(model.transformer.wte(
                    torch.tensor(tokenizer.encode(tokenizer.bos_token), device=device, dtype=torch.long).unsqueeze(0)))
                trigger_embedding_i.requires_grad = True
                trigger_embedding_list.append(trigger_embedding_i)
            trigger_embedding = nn.Parameter(torch.cat(trigger_embedding_list, dim=1))  # bze x n x emb_size
            model.trigger_embedding = trigger_embedding
        elif trigger_format == "key_value":  # learn key values
            trigger_key_values = [(None, None) for _ in range(num_layers)]
            bos_key_values = model(torch.tensor(tokenizer.encode(tokenizer.bos_token), dtype=torch.long).unsqueeze(0).to(device))[
                            "past_key_values"]
            for layer in range(num_layers):
                for i_t in range(num_of_triggers):
                    trigger_i_key_value = copy.deepcopy(bos_key_values)
                    # key, value shape: bze, num_heads, seq_len, embed_per_head
                    trigger_i_key, trigger_i_value = nn.Parameter(trigger_i_key_value[layer][0]), \
                                                     nn.Parameter(trigger_i_key_value[layer][1])

                    trigger_i_key.requires_grad = True
                    trigger_i_value.requires_grad = True

                    # register parameter into optimizer
                    key_name = "l_%d_key_%d" % (layer, i_t)
                    value_name = "l_%d_value_%d" % (layer, i_t)
                    model.register_parameter(name=key_name, param=trigger_i_key)
                    model.register_parameter(name=value_name, param=trigger_i_value)

                    if trigger_key_values[layer][0] is None:
                        trigger_key_values[layer] = (trigger_i_key, trigger_i_value)
                    else:
                        # if multiple triggers
                        trigger_key = torch.cat((trigger_key_values[layer][0], trigger_i_key), dim=-2)
                        trigger_value = torch.cat((trigger_key_values[layer][1], trigger_i_value), dim=-2)
                        trigger_key_values[layer] = (trigger_key, trigger_value)
            trigger_key_values = tuple(trigger_key_values)
        else:
            assert False, "trigger_format: %s not supported" % trigger_format

    if context:  # note: context is [BOS] + context ids
        context_t = torch.tensor(context, device=device, dtype=torch.long)
        while len(context_t.shape) < 2:
            context_t = context_t.unsqueeze(0)
    else:
        assert False, "no context not supported"

    param_optimizer = list(filter(lambda p: p[1].requires_grad, list(model.named_parameters())))

    # debugging
    print("optimizing params: ")
    # print(param_optimizer)
    # get all names
    optimizing_names = []
    for p_o in param_optimizer:
        optimizing_names.append(p_o[0])
    print(optimizing_names)


    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=learning_rate,
                      eps=adam_epsilon)

    lm_bos_output = model(context_t[:, 0])  # BOS

    for i in range(num_iterations):
        past = lm_bos_output["past_key_values"]
        # print("bos past key value")
        # print(past[-1][0][0, 0, :, :20])
        # print(past[-1][1][0, 0, :, :20])
        if num_of_triggers > 0:
            if trigger_format == "token":
                lm_trigger_output = model(inputs_embeds=trigger_embedding, past_key_values=past)
                trigger_key_values = lm_trigger_output["past_key_values"]
            # past = concat_past(past, trigger_key_values, num_layers)

            # print("bos + trigger past key value")
            # print(past[-1][0][0, 0, :, :20])
            # print(past[-1][1][0, 0, :, :20])

        output_so_far = context_t

        context_lm_output = model(context_t[:, 1:-1], past_key_values=trigger_key_values)
        past = context_lm_output["past_key_values"]
        last = output_so_far[:, -1:]

        optimizer.zero_grad()

        # generate conditional prompt
        print("=====Iteration: %d=====" % (i + 1))
        for p_i in range(length):
            lm_output = model(last, past_key_values=past)
            logits, past, all_hidden = (
                lm_output["logits"],  # bze, cur_seq_len, vocab_size
                lm_output["past_key_values"],  # acc_seq_len
                lm_output["hidden_states"],  # num_layers + 1, tuple of (bze, cur_seq_len, hid_sze)
            )
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, top_k)

            for token_idx in set(output_so_far[0].tolist()):
                if logits[0, token_idx] < 0:
                    logits[0, token_idx] *= repetition_penalty
                else:
                    logits[0, token_idx] /= repetition_penalty

            probs = F.softmax(logits, dim=-1)
            if sample:
                last = torch.multinomial(probs, num_samples=1)
            else:
                _, last = torch.topk(probs, k=1, dim=-1)

            output_so_far = torch.cat((output_so_far, last), dim=1)

            # generate one sentence which ends with "." as the prompt
            if last.squeeze(0).data.cpu().numpy()[0] == tokenizer.encode(".")[0]:
                break
        print("context + prompt")
        # print(output_so_far.tolist())
        print(tokenizer.decode(output_so_far.tolist()[0]))
        print("***" * 20)

        # generate response
        response_hidden = []
        response_so_far = None
        first = True
        to_break = False
        for r_i in range(length):
            # TODO: mask trigger key and value

            if num_of_triggers > 0 and not not_mask_trigger:
                # create attention mask
                past_length = past[0][0].shape[-2]
                attention_mask = torch.ones(1, past_length + 1)  # add current 1 to length (also, bze=1)
                attention_mask[0][1: 1 + num_of_triggers] = 0  # bze=1, the first element is BOS
                attention_mask = attention_mask.to(device)
            else:
                attention_mask = None

            # lm_rep_output = model(last, past_key_values=past, attention_mask=attention_mask)
            # debugging
            lm_rep_output = model(last, past_key_values=past, attention_mask=attention_mask, output_attentions=True)

            rep_logits, past, rep_all_hidden = (
                lm_rep_output["logits"],  # bze, cur_seq_len, vocab_size
                lm_rep_output["past_key_values"],  # acc_seq_len
                lm_rep_output["hidden_states"],  # num_layers + 1, tuple of (bze, cur_seq_len, hid_sze)
            )

            rep_logits = rep_logits[:, -1, :] / temperature
            rep_logits = top_k_logits(rep_logits, top_k)
            rep_logits = F.softmax(rep_logits, dim=-1)

            if response_so_far is not None:
                everything_so_far = output_so_far[0].tolist() + response_so_far[0].tolist()
            else:
                everything_so_far = output_so_far[0].tolist()

            for token_idx in set(everything_so_far):
                if rep_logits[0, token_idx] < 0:
                    rep_logits[0, token_idx] *= repetition_penalty
                else:
                    rep_logits[0, token_idx] /= repetition_penalty

            if sample:
                last = torch.multinomial(rep_logits, num_samples=1)
            else:
                _, last = torch.topk(rep_logits, k=1, dim=-1)

            last_hidden = rep_all_hidden[-1]

            if first:
                first = False
            else:
                response_hidden.append(last_hidden)
                # print(tokenizer.decode(last.tolist()[0]))

            if to_break:
                break

            response_so_far = last if response_so_far is None else torch.cat((response_so_far, last), dim=1)

            # generate one sentence which ends with "." as the prompt
            if last.squeeze(0).data.cpu().numpy()[0] == tokenizer.encode(".")[0]:
                to_break = True
                # break
        print("response: ")
        # print(response_so_far.tolist())
        print(tokenizer.decode(response_so_far.tolist()[0]))
        print("***" * 20)

        ce_loss = nn.CrossEntropyLoss()
        prediction = classifier(torch.mean(torch.cat(response_hidden, dim=1), dim=1))
        label = torch.tensor([class_label], device=device, dtype=torch.long)
        discrim_loss = ce_loss(prediction, label)

        # debuggin
        # print("hidden")
        # print(torch.cat(response_hidden, dim=1).shape)
        # print(torch.cat(response_hidden, dim=1))
        # print()
        # print("attn")
        # print(lm_rep_output["attentions"][-1].shape)
        # # print(lm_rep_output["attentions"][-1][0][0])
        # print(lm_rep_output["attentions"][0][0][0])
        # print("past")
        # print("key")  # bze, n_heads, seq, head_dim
        # print(past[-1][0][0][0, :, :5])
        # print("layer 0")
        # print(past[0][0][0][0, :, :5])



        # debugging
        # print("prediction: ")
        # print(prediction)
        print("discrim loss: %.6f" % discrim_loss.data.cpu().numpy())

        if num_of_triggers > 0:
            # compute gradients
            discrim_loss.backward()

            # # debugging: check grad
            # if trigger_format == "token":
            #     print("token grad")
            #     print(trigger_embedding.grad)
            #
            #     # # debugging
            #     # print("original trigger embedding")
            #     # print(trigger_embedding)
            # else:
            #     print("trigger_key_value_grad")
            #     # print(model.l_12_key_0.grad.shape)
            #     # print(model.l_12_key_0.grad)
            #     print(model.l_12_value_0.grad.shape)
            #     print(model.l_12_value_0.grad)

            optimizer.step()

            # # debugging
            # print("new trigger embedding")
            # print(trigger_embedding)

        print("\n\n")


def run_prompt_trigger_example(
    pretrained_model="gpt2-medium",
    cond_text="",
    uncond=False,
    num_samples=1,
    num_of_triggers=1,
    trigger_format="token",
    not_mask_trigger=False,
    learning_rate=5e-5,
    adam_epsilon=1e-8,
    discrim=None,
    class_label=-1,
    length=100,
    temperature=1.0,
    top_k=10,
    sample=False,
    num_iterations=10,
    seed=0,
    no_cuda=False,
    repetition_penalty=1.0,
):
    # set Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # set the device
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"

    # load pretrained model
    model = GPT2LMHeadModel.from_pretrained(pretrained_model, output_hidden_states=True)
    model.to(device)
    model.eval()

    # load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)

    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    # figure out conditioning text
    if uncond:
        tokenized_cond_text = tokenizer.encode([tokenizer.bos_token])
    else:
        raw_text = cond_text
        while not raw_text:
            print("Did you forget to add `--cond_text`? ")
            raw_text = input("Model prompt >>> ")
        tokenized_cond_text = tokenizer.encode(tokenizer.bos_token + raw_text)

    print("= Prefix of sentence =")
    print(tokenizer.decode(tokenized_cond_text))
    print()
    
    generate_prompt(model, tokenizer, num_of_triggers, learning_rate=learning_rate, adam_epsilon=adam_epsilon,
                    context=tokenized_cond_text, device=device, discrim=discrim, class_label=class_label,
                    trigger_format=trigger_format, not_mask_trigger=not_mask_trigger, length=length, 
                    temperature=temperature, top_k=top_k, sample=sample,
                    num_iterations=num_iterations, repetition_penalty=repetition_penalty)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model",
        "-M",
        type=str,
        default="gpt2-medium",
        help="pretrained model name or path to local checkpoint",
    )
    parser.add_argument("--uncond", action="store_true", help="Generate from end-of-text as prefix")
    parser.add_argument("--cond_text", type=str, default="Today is Monday. ", help="Prefix texts to condition on")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate from the modified latents",
    )
    parser.add_argument(
        "--discrim",
        "-D",
        type=str,
        default=None,
        help="Discriminator to use",
    )
    parser.add_argument(
        "--class_label",
        type=int,
        default=-1,
        help="Class label used for the discriminator",
    )
    parser.add_argument("--num_of_triggers", type=int, default=1)
    parser.add_argument(
        "--trigger_format",
        type=str,
        default="token",
        choices=("token", "key_value"),
        help="trigger format to use",
    )
    parser.add_argument("--not_mask_trigger", action="store_true", help="whether to mask the trigger(s) for response generation")

    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--sample", action="store_true", help="Generate from end-of-text as prefix")
    parser.add_argument("--num_iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true", help="no cuda")
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Penalize repetition. More than 1.0 -> less repetition",
    )

    args = parser.parse_args()
    run_prompt_trigger_example(**vars(args))
