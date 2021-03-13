import argparse
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

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


# Gets the score for the top-k logits to improve quality of samples.
def top_k_logits(logits, k):
    if k == 0:
        return logits
    values = torch.topk(logits, k)[0]
    batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
    return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)


def sample_sequence(model, length, past, last, temperature, top_k, repetition_penalty, sample, no_trigger,
                    output_so_far, tokenizer):
    with torch.no_grad():
        response_so_far = None
        two_sentences = False
        prompt = None
        response_hidden = []
        to_break = False
        first = True
        for _ in range(length):
            lm_output = model(last, past_key_values=past)
            logits, past, all_hidden = (
                lm_output["logits"],  # bze, cur_seq_len, vocab_size
                lm_output["past_key_values"],  # acc_seq_len
                lm_output["hidden_states"],  # num_layers + 1, tuple of (bze, cur_seq_len, hid_sze)
            )
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, top_k)

            past_vocab = output_so_far
            if response_so_far is not None:
                past_vocab = past_vocab + response_so_far[0].tolist()
            for token_idx in set(past_vocab):
                if logits[0, token_idx] < 0:
                    logits[0, token_idx] *= repetition_penalty
                else:
                    logits[0, token_idx] /= repetition_penalty

            probs = F.softmax(logits, dim=-1)
            if sample:
                last = torch.multinomial(probs, num_samples=1)
            else:
                _, last = torch.topk(probs, k=1, dim=-1)

            if no_trigger and two_sentences:
                if first:
                    first = False
                else:
                    last_hidden = all_hidden[-1]
                    response_hidden.append(last_hidden)
                    if to_break:
                        break

            response_so_far = last if response_so_far is None else torch.cat((response_so_far, last), dim=1)

            # generate one sentence which ends with "." as the prompt
            if last.squeeze(0).data.cpu().numpy()[0] == tokenizer.encode(".")[0]:
                if no_trigger:
                    if two_sentences:
                        to_break = True
                    else:
                        two_sentences = True
                        print("prompt: %s" % tokenizer.decode(response_so_far.tolist()[0]))
                        prompt = response_so_far.tolist()[0]
                        response_so_far = None
                else:
                    break


        response = tokenizer.decode(response_so_far.tolist()[0])

        print(response)

        return response, prompt, response_hidden


def main(pretrained_model, cond_text, num_samples, discrim, class_label, length, temperature,
         top_k, sample, seed, no_cuda, repetition_penalty, no_trigger, sample_prompt):
    torch.manual_seed(seed)

    # set the device
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"

    # load pretrained model
    model = GPT2LMHeadModel.from_pretrained(pretrained_model, output_hidden_states=True)
    model.to(device)
    model.eval()

    # load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
    context = tokenizer.encode(cond_text)
    output_so_far = context
    context_t = torch.tensor(context, device=device, dtype=torch.long)
    while len(context_t.shape) < 2:
        context_t = context_t.unsqueeze(0)


    if no_trigger and sample_prompt:
        classifier, class_id = get_classifier(discrim, class_label, device)

    with torch.no_grad():
        context_lm_output = model(context_t[:, :-1])
        past = context_lm_output["past_key_values"]
        last = context_t[:, -1:]

    all_responses = list()
    best_loss = 100
    best_prompt = None
    ce_loss = nn.CrossEntropyLoss()
    if sample_prompt:
        classifier, class_id = get_classifier(discrim, class_label, device)

    # get samples from the model
    for i in range(num_samples):
        print("==sample: %d==" % (i + 1))
        response, prompt, response_hidden = sample_sequence(
            model=model, length=length, past=past, last=last, temperature=temperature, top_k=top_k,
            repetition_penalty=repetition_penalty, sample=sample, no_trigger=no_trigger, output_so_far=output_so_far,
            tokenizer=tokenizer)
        all_responses.append(response)

        if sample_prompt:
            prediction = classifier(torch.mean(torch.cat(response_hidden, dim=1), dim=1))
            label = torch.tensor([class_label], device=device, dtype=torch.long)
            discrim_loss = ce_loss(prediction, label)
            if discrim_loss < best_loss:
                best_loss = discrim_loss
                best_prompt = prompt

    if sample_prompt:
        print("***"*20)
        print("generating from the best prompt without trigger prompt input")
        print("best prompt: %s" % tokenizer.decode(best_prompt))
        print("best prompt loss: %.6f" % best_loss)
        all_responses = list()
        context_t = torch.tensor(context + best_prompt, device=device, dtype=torch.long)
        while len(context_t.shape) < 2:
            context_t = context_t.unsqueeze(0)
        with torch.no_grad():
            context_lm_output = model(context_t[:, :-1])
            past = context_lm_output["past_key_values"]
            last = context_t[:, -1:]
        for i in range(num_samples):
            print("==sample: %d==" % (i + 1))
            response, prompt, response_hidden = sample_sequence(
                model=model, length=length, past=past, last=last, temperature=temperature, top_k=top_k,
                repetition_penalty=repetition_penalty, sample=sample, no_trigger=False,
                output_so_far=output_so_far, tokenizer=tokenizer)
            all_responses.append(response)


    # run classifier on all responses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model",
        "-M",
        type=str,
        default="gpt2-medium",
        help="pretrained model name or path to local checkpoint",
    )
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
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--sample", action="store_true", help="Generate from end-of-text as prefix")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true", help="no cuda")
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Penalize repetition. More than 1.0 -> less repetition",
    )
    parser.add_argument("--no_trigger", action="store_true", help="if no trigger: generate two sentences.")
    parser.add_argument("--sample_prompt", action="store_true", help="if True, then use a classifier to choose the "
                                                                     "best prompt and generate num_samples responses "
                                                                     "accordingly, similar to the proposed method")

    args = parser.parse_args()

    main(**vars(args))
