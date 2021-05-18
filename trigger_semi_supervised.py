"""
CUDA_VISIBLE_DEVICES=1 python trigger_semi_supervised.py --pretrained_model gpt2-medium --discrim sentiment --class_label 3 --train_filename persona_train.txt --eval_filename persona_eval.txt --do_train --num_of_triggers 1 --trigger_format key_value --num_epochs 2 --num_iterations 2 --batch_size 4 --verbose --check_real_loss --eval_num_per_cond 2 --multiple_input --reset_pos_emb --length 40 --sample --temperature 1.0 --top_k 10 --gumbel_softmax --gumbel_temperature 1.0 --detach --repetition_penalty 1.0 --learning_rate 5e-3 --gradient_accumulation_steps 1 --seed 0

"""

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

from utils import get_classifier, generate_next, concat_past, expand_past, read_file

TRIGGER_POSITION_ID = 0  # for position reset


def penalize_new_line(logits, block_list):
    for b_i in range(logits.shape[0]):
        for nt in block_list:
            if logits[b_i, -1, nt] < 0:
                logits[b_i, -1, nt] *= 5
            else:
                logits[b_i, -1, nt] /= 5
    return logits


# initialize trigger
# Note: since we use the same trigger for all inputs in a batch, we only create/register trigger(s)
# for one and repeat it
def init_trigger(model, tokenizer, num_of_triggers, trigger_format, num_layers, device):
    if num_of_triggers > 0:
        if trigger_format == "token":  # learn a continuous embedding
            trigger_embedding_list = []
            for _ in range(num_of_triggers):
                trigger_embedding_i = copy.deepcopy(model.transformer.wte(
                    torch.tensor(tokenizer.encode(tokenizer.bos_token), device=device, dtype=torch.long).unsqueeze(0)))
                trigger_embedding_list.append(trigger_embedding_i)
            ori_trigger_embedding = nn.Parameter(torch.cat(trigger_embedding_list, dim=1))  # bze x n x emb_size
            model.ori_trigger_embedding = ori_trigger_embedding  # register to the model (optimizer)
            # trigger_embedding = trigger_embedding.repeat(batch_size, 1, 1)  # cannot do it here, otherwise
            # trigger_embedding becomes a non-leaf node where the grad will not backprop
        elif trigger_format == "key_value":  # learn key values
            ori_trigger_key_values = [(None, None) for _ in range(num_layers)]
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

                    if ori_trigger_key_values[layer][0] is None:
                        ori_trigger_key_values[layer] = (trigger_i_key, trigger_i_value)
                    else:
                        # if multiple triggers
                        trigger_key = nn.Parameter(torch.cat((ori_trigger_key_values[layer][0], trigger_i_key), dim=-2))
                        trigger_value = nn.Parameter(torch.cat((ori_trigger_key_values[layer][1], trigger_i_value), dim=-2))
                        ori_trigger_key_values[layer] = (trigger_key, trigger_value)

                # register parameter into optimizer
                key_name = "l_%d_key" % layer
                value_name = "l_%d_value" % layer
                if num_of_triggers == 1:
                    model.register_parameter(name=key_name, param=trigger_i_key)
                    model.register_parameter(name=value_name, param=trigger_i_value)
                else:
                    model.register_parameter(name=key_name, param=trigger_key)
                    model.register_parameter(name=value_name, param=trigger_value)

            ori_trigger_key_values = tuple(ori_trigger_key_values)
            model.ori_trigger_key_values = ori_trigger_key_values
            # trigger_key_values = expand_past(trigger_key_values, num_layers, batch_size)  # similar to
            # trigger_embedding, need leaf level grad
        else:
            assert False, "trigger_format: %s not supported" % trigger_format


def prep_inputs(batch_cond_text_list, tokenizer, device, t_pad_token):
    batch_max_length = 0
    batch_min_length = 10000
    batch_input_ids = list()
    all_inputs, all_attention_masks, all_lengths, all_padding_length = list(), list(), list(), list()
    padding_token = tokenizer.encode(t_pad_token)[0]  # WARNING: BOS is for GPT2 only. Should use padding token
    for cond_text in batch_cond_text_list:
        inputs_ids = tokenizer.encode(tokenizer.bos_token + cond_text)
        batch_max_length = len(inputs_ids) if len(inputs_ids) > batch_max_length else batch_max_length
        batch_min_length = len(inputs_ids) if len(inputs_ids) < batch_min_length else batch_min_length
        batch_input_ids.append(inputs_ids)
    for inputs_ids in batch_input_ids:
        all_lengths.append(len(inputs_ids))
        padding_len = batch_max_length - len(inputs_ids)
        attention_mask = [1] * len(inputs_ids) + [0] * padding_len
        inputs_ids = inputs_ids + [padding_token] * padding_len

        all_padding_length.append(padding_len)
        all_inputs.append(inputs_ids)
        all_attention_masks.append(attention_mask)

    all_input_ids = torch.tensor(all_inputs, dtype=torch.long, device=device)
    all_attention_masks = torch.tensor(all_attention_masks, dtype=torch.long, device=device)

    return all_input_ids, all_attention_masks, batch_min_length, batch_max_length, all_lengths


def generate_prompt_response(model, tokenizer, mode, lm_bos_output, batch_size, device, class_label,
                             num_iterations, gradient_accumulation_steps, sample, gumbel_softmax,
                             detach, reset_pos_emb, not_mask_trigger, gumbel_temperature, top_k, temperature,
                             repetition_penalty, max_grad_norm, length, t_pad_token, stop_token,
                             num_epochs, context_list, verbose, check_real_loss,
                             seed, optimizer, classifier, ce_loss, num_layers, block_list,
                             num_of_triggers, trigger_format, shuffle_data, print_real,
                             ):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if mode == "eval":
        num_epochs = 1

    total_data_batches = len(context_list) // batch_size

    for epoch in range(num_epochs):
        print("&&&&& epoch: %d &&&&&" % (epoch + 1))

        if mode == "train" and shuffle_data:
            random.shuffle(context_list)

        epoch_loss = 0

        for cond_list_idx in range(total_data_batches):
            cond_list = context_list[cond_list_idx * batch_size: (cond_list_idx + 1) * batch_size]
            all_input_ids, all_attention_masks, batch_min_length, batch_max_length, all_lengths = prep_inputs(cond_list,
                                                                                                              tokenizer,
                                                                                                              device,
                                                                                                              t_pad_token)

            model.zero_grad()

            loss_per_update = 0
            total_loss = 0

            if reset_pos_emb:
                c_position_ids = torch.arange(1, batch_min_length - 1, dtype=torch.long, device=device)
                c_position_ids = c_position_ids.unsqueeze(0).repeat(batch_size, 1)
            else:
                c_position_ids = None

            if reset_pos_emb:
                t_position_ids = torch.ones(batch_size, num_of_triggers).to(torch.long).to(device) * TRIGGER_POSITION_ID
            else:
                t_position_ids = None

            all_context_prompts = []
            all_generated_prompt_length = []

            for i in range(num_iterations):

                past = lm_bos_output["past_key_values"]

                if num_of_triggers > 0:
                    if trigger_format == "token":
                        trigger_embedding = model.ori_trigger_embedding.repeat(batch_size, 1, 1)
                        lm_trigger_output = model(inputs_embeds=trigger_embedding, position_ids=t_position_ids)
                        trigger_key_values = lm_trigger_output["past_key_values"]
                    else:
                        trigger_key_values = expand_past(model.ori_trigger_key_values, num_layers, batch_size)
                    past = concat_past(past, trigger_key_values, num_layers)
                else:
                    trigger_key_values = None

                output_so_far = all_input_ids[:, :batch_min_length]  # bze x (batch_min_length - 1)

                context_lm_output = model(all_input_ids[:, 1: batch_min_length - 1], past_key_values=past,
                                          position_ids=c_position_ids, )

                past = context_lm_output["past_key_values"]
                last = output_so_far[:, batch_min_length - 1: batch_min_length]

                gumbel_vector = None
                if detach:
                    all_gumbel_vectors = None

                prompt_not_done = torch.ones(batch_size, 1, dtype=torch.uint8, device=device)
                generated_prompt_length = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
                prompt_stop_first = True

                # generate conditional prompt
                if verbose:
                    print("=====Epoch: %d; data_batch: %d; Iteration: %d=====" % (epoch + 1, cond_list_idx + 1, i + 1))

                for p_i in range(length):
                    if reset_pos_emb:
                        past_length = past[0][0].size(-2)
                        p_position_ids = torch.arange(past_length - num_of_triggers, past_length - num_of_triggers + 1,
                                                      dtype=torch.long, device=device)
                        p_position_ids = p_position_ids.unsqueeze(0).repeat(batch_size, 1)
                    else:
                        p_position_ids = None
                    if gumbel_softmax and gumbel_vector is not None:
                        last_emb = torch.mm(gumbel_vector, model.transformer.wte.weight).unsqueeze(
                            1)  # needs to be bze, n, emb
                        lm_output = model(inputs_embeds=last_emb, past_key_values=past, position_ids=p_position_ids)
                    else:
                        lm_output = model(last, past_key_values=past, position_ids=p_position_ids)

                    logits, past, all_hidden = (
                        lm_output["logits"],  # bze, cur_seq_len, vocab_size
                        lm_output["past_key_values"],  # acc_seq_len
                        lm_output["hidden_states"],  # num_layers + 1, tuple of (bze, cur_seq_len, hid_sze)
                    )

                    vocab_size = logits.shape[-1]

                    logits = penalize_new_line(logits, block_list)

                    # last: bze x 1; gumbel_vector: bze x vocab_size
                    last, gumbel_vector = generate_next(logits, output_so_far, top_k=top_k, temperature=temperature,
                                                        repetition_penalty=repetition_penalty, sample=sample,
                                                        gumbel_softmax=gumbel_softmax,
                                                        gumbel_temperature=gumbel_temperature, detach=detach)
                    # manually assign end token is too long
                    if p_i == length - 1:
                        for m_b_i in range(batch_size):
                            if generated_prompt_length[m_b_i] == 0:
                                last[m_b_i] = tokenizer.encode(stop_token)[0]  # encode outputs a list (1 element)
                                if gumbel_softmax:
                                    gumbel_vector[m_b_i] = F.one_hot(
                                        torch.tensor(tokenizer.encode(stop_token), dtype=torch.long, device=device),
                                        num_classes=vocab_size)

                    # double check the length (p_i vs. lengths below)
                    is_generated = torch.tensor(all_lengths, device=device).unsqueeze(-1) <= (
                                p_i + batch_min_length)  # bze x 1. is generated or stil in the context
                    is_end_token = last == torch.tensor(tokenizer.encode(stop_token), device=device)  # bze x 1
                    is_actually_ending = is_generated * is_end_token

                    # keep track of prompt length
                    generated_prompt_length = generated_prompt_length + prompt_not_done * is_actually_ending * p_i

                    # if generated, use the generated token as last; otherwise (from the original), copy the
                    # orignal token/gumbel_vector
                    if batch_min_length + p_i < all_input_ids.shape[1]:
                        last = last * is_generated + all_input_ids[:, batch_min_length + p_i].unsqueeze(1) * (
                            ~is_generated)  # is_generated is bool. need to use "~" instead of (1-is_generated)
                    else:
                        last = last

                    if gumbel_softmax:
                        if batch_min_length + p_i < all_input_ids.shape[1]:
                            ori_one_hot = F.one_hot(all_input_ids[:, batch_min_length + p_i],
                                                    num_classes=vocab_size)  # bze x vocab_size
                            gumbel_vector = gumbel_vector * is_generated + ori_one_hot * (~is_generated)
                        else:
                            gumbel_vector = gumbel_vector
                    if prompt_stop_first and torch.sum(is_actually_ending) > 0:
                        prompt_stop_first = False
                        min_p_past = past
                        min_p_last = last
                        min_gumbel_vector = gumbel_vector

                    if detach:
                        # WARNING: This can be quite large
                        if all_gumbel_vectors is None:
                            all_gumbel_vectors = gumbel_vector.unsqueeze(1)  # bze x 1 x vocab_size
                        else:
                            all_gumbel_vectors = torch.cat((all_gumbel_vectors, gumbel_vector.unsqueeze(1)),
                                                           dim=1)  # bze x n x vocab_size

                    output_so_far = torch.cat((output_so_far, last), dim=1)  # bze x length

                    prompt_not_done = prompt_not_done * (~is_actually_ending)  # to check is we need to stop by summing

                    if torch.sum(prompt_not_done) == 0:
                        break

                if verbose:
                    print("context + prompt")
                    batch_context_prompts = []
                    for cp_i in range(batch_size):
                        cp_i_text = tokenizer.decode(output_so_far[cp_i][:(
                                    generated_prompt_length[cp_i].item() + 1 + batch_min_length)].tolist())
                        print(cp_i_text)
                        batch_context_prompts.append(cp_i_text)
                    print("***" * 20)
                else:
                    batch_context_prompts = []
                    for cp_i in range(batch_size):
                        cp_i_text = tokenizer.decode(output_so_far[cp_i][:(
                                generated_prompt_length[cp_i].item() + 1 + batch_min_length)].tolist())
                        batch_context_prompts.append(cp_i_text)

                all_context_prompts.append(batch_context_prompts)
                all_generated_prompt_length.append(generated_prompt_length + batch_min_length + 1)

                if mode == "eval":  # only need context and prompts for evaluation
                    continue

                cp_min_length = torch.min(generated_prompt_length.squeeze(1)) + batch_min_length + 1

                if detach:
                    detach_context_output = model(all_input_ids[:, :batch_min_length])
                    past = detach_context_output["past_key_values"]
                    all_gumbel_embeddings = torch.matmul(
                        all_gumbel_vectors[:, :cp_min_length - batch_min_length - 1, :], model.transformer.wte.weight)
                    last_detach_output = model(inputs_embeds=all_gumbel_embeddings, past_key_values=past)
                    past = last_detach_output["past_key_values"]

                    gumbel_vector = min_gumbel_vector
                else:
                    # adjust past, last, and gumbel_vector
                    past = min_p_past
                    last = min_p_last

                everything_so_far = output_so_far[:, :cp_min_length]

                # generate response
                response_hidden = None
                to_break = False

                response_not_done = torch.ones(batch_size, 1, dtype=torch.uint8, device=device)
                generated_response_length = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

                for r_i in range(length):
                    # TODO: mask trigger key and value
                    if num_of_triggers > 0 and not not_mask_trigger and not detach:
                        # create attention mask
                        past_length = past[0][0].shape[-2]
                        attention_mask = torch.ones(batch_size, past_length + 1)  # add current 1 to length
                        attention_mask[:, 1: 1 + num_of_triggers] = 0  # bze=1, the first element is BOS
                        attention_mask = attention_mask.to(device)
                    else:
                        attention_mask = None

                    if num_of_triggers > 0 and reset_pos_emb and not detach:
                        past_length = past[0][0].size(-2)
                        r_position_ids = torch.arange(past_length - num_of_triggers, past_length - num_of_triggers + 1,
                                                      dtype=torch.long, device=device)
                        r_position_ids = r_position_ids.unsqueeze(0).repeat(batch_size, 1)
                    else:
                        r_position_ids = None

                    # debugging
                    if gumbel_softmax:
                        last_emb = torch.mm(gumbel_vector, model.transformer.wte.weight).unsqueeze(
                            1)  # bze x 1 x hidden
                        lm_rep_output = model(inputs_embeds=last_emb, past_key_values=past,
                                              attention_mask=attention_mask,
                                              output_attentions=True, position_ids=r_position_ids)
                    else:
                        lm_rep_output = model(last, past_key_values=past, attention_mask=attention_mask,
                                              output_attentions=True,
                                              position_ids=r_position_ids)

                    rep_logits, past, rep_all_hidden = (
                        lm_rep_output["logits"],  # bze, cur_seq_len, vocab_size
                        lm_rep_output["past_key_values"],  # acc_seq_len
                        lm_rep_output["hidden_states"],  # num_layers + 1, tuple of (bze, cur_seq_len, hid_sze)
                    )

                    rep_logits = penalize_new_line(rep_logits, block_list)

                    last, gumbel_vector = generate_next(rep_logits, everything_so_far, top_k=top_k,
                                                        temperature=temperature,
                                                        repetition_penalty=repetition_penalty, sample=sample,
                                                        gumbel_softmax=gumbel_softmax,
                                                        gumbel_temperature=gumbel_temperature, detach=detach)

                    last_hidden = rep_all_hidden[-1]

                    if response_hidden is None:
                        response_hidden = last_hidden
                    else:
                        response_hidden = torch.cat((response_hidden, last_hidden), dim=1)  # bze, n, hid_size

                    if to_break:
                        break

                    # manually assign end token is too long
                    if r_i == length - 1:
                        for r_m_b_i in range(batch_size):
                            if generated_response_length[r_m_b_i] == 0:
                                last[r_m_b_i] = tokenizer.encode(stop_token)[0]  # encode outputs a list (1 element)
                                gumbel_vector[r_m_b_i] = F.one_hot(
                                    torch.tensor(tokenizer.encode(stop_token), dtype=torch.long, device=device),
                                    num_classes=vocab_size)

                    # adjust
                    r_is_generated = generated_prompt_length + 1 + batch_min_length <= (
                                r_i + cp_min_length)  # bze x 1. is generated or stil in the context+prompt
                    r_is_end_token = last == torch.tensor(tokenizer.encode(stop_token), device=device)  # bze x 1
                    r_is_actually_ending = r_is_generated * r_is_end_token

                    # keep track of response length
                    generated_response_length = generated_response_length + response_not_done * r_is_actually_ending * r_i

                    # if generated, use the generated token as last; otherwise (from context+prompt), copy the
                    # orignal token/gumbel_vector
                    if cp_min_length + r_i < output_so_far.shape[1]:
                        last = last * r_is_generated + output_so_far[:, cp_min_length + r_i].unsqueeze(1) * (
                            ~r_is_generated)
                    else:
                        last = last

                    if gumbel_softmax:
                        if cp_min_length - batch_min_length + r_i < all_gumbel_vectors.shape[1]:
                            gumbel_vector = gumbel_vector * r_is_generated + all_gumbel_vectors[:,
                                                                             cp_min_length - batch_min_length + r_i,
                                                                             :] * (~r_is_generated)
                        else:
                            gumbel_vector = gumbel_vector

                    everything_so_far = torch.cat((everything_so_far, last), dim=1)

                    response_not_done = response_not_done * (
                        ~r_is_actually_ending)  # to check is we need to stop by summing
                    if torch.sum(response_not_done) == 0:
                        to_break = True

                if verbose:
                    print("response: ")
                    for pr_i in range(batch_size):
                        if generated_response_length[pr_i] == 0:
                            print(tokenizer.decode(everything_so_far[pr_i][generated_prompt_length[
                                                                               pr_i].item() + 1 + batch_min_length:].tolist()))  # exceeds the max length set (so generated_prompt_length is 0)
                        else:
                            print(tokenizer.decode(everything_so_far[pr_i][
                                                   generated_prompt_length[pr_i].item() + 1 + batch_min_length:(
                                                               generated_response_length[
                                                                   pr_i].item() + 1 + cp_min_length)].tolist()))
                    print("***" * 20)

                extracted_hidden = None
                # hidden: bze, 1, hid_size
                for hb_i in range(batch_size):
                    hb_i_start = generated_prompt_length[hb_i] + 1 + batch_min_length - cp_min_length + 1
                    hb_i_end = generated_response_length[hb_i] + 1 + 1
                    hb_i_hidden = torch.mean(response_hidden[hb_i:hb_i + 1, hb_i_start:hb_i_end, :],
                                             dim=1)  # 1, hid_size
                    if extracted_hidden is None:
                        extracted_hidden = hb_i_hidden
                    else:
                        extracted_hidden = torch.cat((extracted_hidden, hb_i_hidden), dim=0)

                prediction = classifier(extracted_hidden)
                label = torch.tensor([class_label], device=device, dtype=torch.long).repeat(batch_size)
                discrim_loss = ce_loss(prediction, label)

                if verbose:
                    print("discrim loss: %.6f" % discrim_loss.data.cpu().numpy())
                loss_per_update += discrim_loss.item()

                if num_of_triggers > 0:
                    # compute gradients
                    discrim_loss.backward()

                    # # debugging: check grad
                    if trigger_format == "token":
                        print_debug = False
                        if print_debug:
                            print("token grad")
                            print(model.ori_trigger_embedding.grad)
                            #             print(trigger_embedding)
                            #
                            # debugging
                            print("original trigger embedding")
                            print(model.ori_trigger_embedding)
                    # else:
                    #     print("trigger_key_value_grad")
                    #     # print(model.l_12_key_0.grad.shape)
                    #     # print(model.l_12_key_0.grad)
                    #     print(model.l_12_value.grad.shape)
                    #     # print(model.l_12_value_1.grad)
                    #     print(model.l_12_value)

                    if (i + 1) % gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                        optimizer.step()
                        model.zero_grad()
                        if verbose:
                            print("\n=======update loss: %.6f=======" % (loss_per_update / gradient_accumulation_steps))
                        total_loss += loss_per_update
                        loss_per_update = 0

                    # # debugging
                    # print("new trigger embedding")
                    # print(trigger_embedding)

                if verbose:
                    print("\n\n")

            if mode == "train":
                print("epoch: %d; data batch: %d/%d; total training average loss: %.6f\n" % (
                epoch + 1, cond_list_idx + 1, total_data_batches, total_loss / num_iterations))

            if check_real_loss:
                data_batch_loss = sample_gpt2(model, tokenizer, all_context_prompts, all_generated_prompt_length,
                                              t_pad_token, classifier, class_label, ce_loss, block_list, device,
                                              length, top_k, temperature, repetition_penalty, sample, stop_token,
                                              print_real, mode,
                                              )
                epoch_loss += data_batch_loss

        if check_real_loss:
            print("epoch %d: total real loss: %.6f\n\n" % (epoch + 1, epoch_loss / total_data_batches))
            print("<<<<<<>>>>>>\n\n")


def sample_gpt2(model, tokenizer, all_context_prompts, all_generated_prompt_length,
                t_pad_token, classifier, class_label, ce_loss, block_list, device,
                length, top_k, temperature, repetition_penalty, sample, stop_token,
                print_real, mode):
    real_loss_all = 0
    padding_token = tokenizer.encode(t_pad_token)[0]

    for real_i in range(len(all_context_prompts)):
        cp_length_i = all_generated_prompt_length[real_i]
        min_cp_length_i = torch.min(cp_length_i)  # tensor of float (no shape)
        max_cp_length_i = torch.max(cp_length_i)
        cp_i_text = all_context_prompts[real_i]

        cp_i_input_ids = list()
        for cp_i_j in cp_i_text:
            input_ids = tokenizer.encode(cp_i_j)
            padding_len = max_cp_length_i - len(input_ids)
            input_ids = input_ids + [padding_token] * padding_len
            cp_i_input_ids.append(input_ids)

        # note: for instance, '\n\n' will be decoded as 628 in gpt2, but it may be encoded as '198 198' if it's
        # connected to another token (e.g. '\n\n-'), which creates a problem in length
        ck_max_cp_length_i = max(len(cp_i_inp) for cp_i_inp in cp_i_input_ids)
        if ck_max_cp_length_i > max_cp_length_i:
            max_cp_length_i = ck_max_cp_length_i
            for cp_i_inp_idx in range(len(cp_i_input_ids)):
                if len(cp_i_input_ids[cp_i_inp_idx]) < ck_max_cp_length_i:
                    cp_i_input_ids[cp_i_inp_idx] += (max_cp_length_i - len(cp_i_input_ids[cp_i_inp_idx])) * [
                        padding_token]
                elif len(cp_i_input_ids[cp_i_inp_idx]) == ck_max_cp_length_i:
                    cp_length_i[cp_i_inp_idx] = ck_max_cp_length_i

        cp_i_inputs = torch.tensor(cp_i_input_ids, dtype=torch.long, device=device)

        cp_i_batch_size = len(cp_i_text)
        real_response_hidden = None
        real_rp_h_ck = None
        to_break = False
        cp_i_response_not_done = torch.ones(cp_i_batch_size, 1, dtype=torch.uint8, device=device)
        cp_i_generated_response_length = torch.zeros(cp_i_batch_size, 1, dtype=torch.long, device=device)

        with torch.no_grad():  # Need this?
            cp_i_so_far = cp_i_inputs[:, :min_cp_length_i]  # bze x (min_cp_length_i - 1)
            cp_i_context_lm_output = model(cp_i_inputs[:, :min_cp_length_i - 1])
            past = cp_i_context_lm_output["past_key_values"]
            last = cp_i_inputs[:, min_cp_length_i - 1: min_cp_length_i]

            for rr_i in range(length):  # length + max_cp_length_i - min_cp_length_i?
                lm_cp_i_output = model(last, past_key_values=past)
                cp_i_logits, past, cp_i_hidden = (lm_cp_i_output["logits"],  # bze, cur_seq_len, vocab_size
                                                  lm_cp_i_output["past_key_values"],  # acc_seq_len
                                                  lm_cp_i_output[
                                                      "hidden_states"])  # num_layers + 1, tuple of (bze, cur_seq_len, hid_sze)

                cp_i_logits = penalize_new_line(cp_i_logits, block_list)

                last, _ = generate_next(cp_i_logits, cp_i_so_far, top_k=top_k, temperature=temperature,
                                        repetition_penalty=repetition_penalty, sample=sample,
                                        gumbel_softmax=False, gumbel_temperature=1.0, detach=False)

                cp_i_last_hidden = cp_i_hidden[-1]
                if real_response_hidden is None:
                    real_response_hidden = cp_i_last_hidden
                    real_rp_h_ck = last
                else:
                    real_response_hidden = torch.cat((real_response_hidden, cp_i_last_hidden),
                                                     dim=1)  # bze, n, hid_size
                    real_rp_h_ck = torch.cat((real_rp_h_ck, last), dim=1)  # bze, n
                if to_break:
                    break

                # manually assign end token if too long
                if rr_i == length - 1:
                    for rr_b_i in range(cp_i_batch_size):
                        if cp_i_generated_response_length[rr_b_i] == 0:
                            last[rr_b_i] = tokenizer.encode(stop_token)[0]  # encode outputs a list (1 element)

                # adjust
                rr_i_is_generated = cp_length_i <= (
                            rr_i + min_cp_length_i)  # bze x 1. is generated or stil in the context+prompt
                rr_i_is_end_token = last == torch.tensor(tokenizer.encode(stop_token), device=device)  # bze x 1
                rr_i_is_actually_ending = rr_i_is_generated * rr_i_is_end_token

                # keep track of response length
                cp_i_generated_response_length = cp_i_generated_response_length + cp_i_response_not_done * rr_i_is_actually_ending * rr_i

                # if generated, use the generated token as last; otherwise (from context+prompt),
                # copy the orignal token/gumbel_vector
                if min_cp_length_i + rr_i < max_cp_length_i:
                    last = last * rr_i_is_generated + cp_i_inputs[:, min_cp_length_i + rr_i].unsqueeze(1) * (
                        ~rr_i_is_generated)
                else:
                    last = last

                cp_i_so_far = torch.cat((cp_i_so_far, last), dim=1)

                cp_i_response_not_done = cp_i_response_not_done * (~rr_i_is_actually_ending)
                if torch.sum(cp_i_response_not_done) == 0:
                    to_break = True

            if print_real or mode == "eval":
                print("Real responses: ")
                for r_pr_i in range(cp_i_batch_size):
                    if cp_i_generated_response_length[r_pr_i] == 0:
                        print(tokenizer.decode(cp_i_so_far[r_pr_i,
                                               :].tolist()))  # exceeds the max length set (so generated_prompt_length is 0)
                    else:
                        print(tokenizer.decode(cp_i_so_far[r_pr_i, :(
                                    cp_i_generated_response_length[r_pr_i].item() + 1 + min_cp_length_i)].tolist()))
                print("***" * 20)

            cp_i_extracted_hidden = None
            for r_hb_i in range(cp_i_batch_size):
                r_hb_i_start = cp_length_i[r_hb_i] - min_cp_length_i + 1
                r_hb_i_end = cp_i_generated_response_length[r_hb_i] + 1 + 1
                r_hb_i_hidden = torch.mean(real_response_hidden[r_hb_i:r_hb_i + 1, r_hb_i_start:r_hb_i_end, :], dim=1)
                if cp_i_extracted_hidden is None:
                    cp_i_extracted_hidden = r_hb_i_hidden
                else:
                    cp_i_extracted_hidden = torch.cat((cp_i_extracted_hidden, r_hb_i_hidden), dim=0)

            prediction = classifier(cp_i_extracted_hidden)
            label = torch.tensor([class_label], device=device, dtype=torch.long).repeat(cp_i_batch_size)
            loss = ce_loss(prediction, label)

            real_loss_all += loss.item()

            if print_real or mode == "eval":
                print("loss: %.6f" % loss.item())
                print()

    print("avg real loss: %.6f\n" % (real_loss_all / len(all_context_prompts)))

    return real_loss_all / len(all_context_prompts)


def main(pretrained_model, discrim, class_label, train_filename, eval_filename, do_baseline, do_train, do_eval,
         num_of_triggers, trigger_format, num_epochs, num_iterations, batch_size, verbose, print_real, check_real_loss, eval_num_per_cond, shuffle_data,
         multiple_input, cond_text, not_mask_trigger, reset_pos_emb,
         length, sample, temperature, top_k, gumbel_softmax, gumbel_temperature, detach, repetition_penalty,
         learning_rate, adam_epsilon, gradient_accumulation_steps, max_grad_norm,
         seed, no_cuda):

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"

    # load pretrained model
    model = GPT2LMHeadModel.from_pretrained(pretrained_model, output_hidden_states=True)
    model.to(device)
    model.eval()  # do not need batchnorm or dropout layers for training/eval

    # load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)

    # WARNING: GPT2 only
    new_line_idx = 198  # '\n'
    new_line_idx_1 = 628  # '\n\n'
    stop_token = "."
    t_pad_token = tokenizer.bos_token
    block_list = [new_line_idx, new_line_idx_1]

    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    classifier, class_label = get_classifier(discrim, class_label, device)

    num_layers = model.config.n_layer

    ce_loss = nn.CrossEntropyLoss()

    lm_bos_output = model(
        torch.tensor(tokenizer.encode(tokenizer.bos_token), dtype=torch.long, device=device).unsqueeze(0).repeat(
            batch_size, 1))  # BOS

    init_trigger(model, tokenizer, num_of_triggers, trigger_format, num_layers, device)

    # optimizer
    param_optimizer = list(filter(lambda p: p[1].requires_grad, list(model.named_parameters())))

    # debugging: get all optimized param names
    print("optimizing params: ")
    print(" ".join(o[0] for o in param_optimizer))

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

    fixed_optimizer = AdamW(optimizer_grouped_parameters,
                      lr=0,
                      eps=adam_epsilon)

    # run
    if not multiple_input:
        train_cond_list = [cond_text] * batch_size
        eval_cond_list = [cond_text] * batch_size
    else:
        train_cond_list = read_file(train_filename)
        eval_cond_list = read_file(eval_filename)

    # baseline:
    if do_baseline:
        print("=======getting baselines=======")
        generate_prompt_response(model, tokenizer, "eval", lm_bos_output, batch_size, device, class_label,
                                 eval_num_per_cond, gradient_accumulation_steps, sample, False,
                                 False, reset_pos_emb, not_mask_trigger, gumbel_temperature, top_k, temperature,
                                 repetition_penalty, max_grad_norm, length, t_pad_token, stop_token,
                                 1, eval_cond_list, verbose, True,
                                 seed, fixed_optimizer, classifier, ce_loss, num_layers, block_list,
                                 num_of_triggers, trigger_format, shuffle_data, print_real,
                                )
        print("\n\n\n")

    # train:
    if do_train:
        print("=======training=======")
        generate_prompt_response(model, tokenizer, "train", lm_bos_output, batch_size, device, class_label,
                                 num_iterations, gradient_accumulation_steps, sample, gumbel_softmax,
                                 detach, reset_pos_emb, not_mask_trigger, gumbel_temperature, top_k, temperature,
                                 repetition_penalty, max_grad_norm, length, t_pad_token, stop_token,
                                 num_epochs, train_cond_list, verbose, check_real_loss,
                                 seed, optimizer, classifier, ce_loss, num_layers, block_list,
                                 num_of_triggers, trigger_format, shuffle_data, print_real,
                                 )
        print("\n\n\n")

    # eval:
    if do_eval:
        print("=======evaluation=======")
        generate_prompt_response(model, tokenizer, "eval", lm_bos_output, batch_size, device, class_label,
                                 eval_num_per_cond, gradient_accumulation_steps, sample, False,
                                 False, reset_pos_emb, not_mask_trigger, gumbel_temperature, top_k, temperature,
                                 repetition_penalty, max_grad_norm, length, t_pad_token, stop_token,
                                 1, eval_cond_list, verbose, True,
                                 seed, fixed_optimizer, classifier, ce_loss, num_layers, block_list,
                                 num_of_triggers, trigger_format, shuffle_data, print_real,
                                 )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    """task"""
    parser.add_argument("--pretrained_model", type=str, default="gpt2-medium",
                        help="pretrained model name or path to local checkpoint")
    parser.add_argument("--discrim", type=str, default=None, help="Discriminator to use",)

    # sentiment: class_label = ["positive", "negative", "very position", "very negative", "neutral"]
    parser.add_argument("--class_label", default=-1, help="Class label used for the discriminator")
    parser.add_argument("--train_filename", type=str, default=None, help="context file for training",)
    parser.add_argument("--eval_filename", type=str, default=None, help="context file for evaluation",)
    parser.add_argument("--do_baseline", action="store_true", help="run for baseline on eval")
    parser.add_argument("--do_train", action="store_true", help="run for training")
    parser.add_argument("--do_eval", action="store_true", help="run for evaluation")

    """training"""
    parser.add_argument("--num_of_triggers", type=int, default=1)
    parser.add_argument("--trigger_format", type=str, default="token", choices=("token", "key_value"),
                        help="trigger format to use")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("--num_iterations", type=int, default=1,
                        help="Number of iterations for sampling during either training or evaluation",)
    parser.add_argument("--batch_size", type=int, default=4, help="batch size for generating prompts/responses")
    parser.add_argument("--verbose", action="store_true", help="log all generated texts (including the "
                                                               "prompts generated with triggers during training")
    parser.add_argument("--print_real", action="store_true", help="whether to log real generation during training")
    parser.add_argument("--check_real_loss", action="store_true", help="check real loss of the generated prompt "
                                                                       "(without trigger) during training")
    parser.add_argument("--eval_num_per_cond", type=int, default=2, help="Number of examples for "
                                                                         "each context during evaluation")
    parser.add_argument("--shuffle_data", action="store_true", help="shuffle training data")

    """trigger"""
    parser.add_argument("--multiple_input", action="store_true",
                        help="whether to use multiple input during training. If not, "
                             "require single input text which will be training in batch")
    parser.add_argument("--cond_text", type=str, default="Today is Monday. ", help="Prefix texts to condition on")
    parser.add_argument("--not_mask_trigger", action="store_true",
                        help="whether to mask the trigger(s) for response generation")
    parser.add_argument("--reset_pos_emb", action="store_true",
                        help="If True, then set position index of the trigger to be TRIGGER_POSITION_ID")

    """decoding"""
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--sample", action="store_true", help="Generate from end-of-text as prefix")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--gumbel_softmax", action="store_true",
                        help="If True, use gumbel softmax instead of argmax in sampling which will "
                             "make it differentiable")
    parser.add_argument("--gumbel_temperature", type=float, default=1.0)
    parser.add_argument("--detach", action="store_true",
                        help="If True, remove trigger when generating responses. It requires using gumbel softmax")
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Penalize repetition. More than 1.0 -> less repetition",
    )

    """optimization"""
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="if > 1, accumulate multiple steps before backprop")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    """general"""
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true", help="no cuda")

    args = parser.parse_args()
    main(**vars(args))




