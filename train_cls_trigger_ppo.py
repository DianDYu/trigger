import copy
import random

import wandb
import time
import os
import numpy as np
from random import choices
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, BlenderbotConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.generation_beam_search import BeamSearchScorer

from ppo_model_ac import BlenderWithValueModel

from ppo import AdaptiveKLController, FixedKLController
from ppo_utils import build_bert_batch_from_txt, logprobs_from_logits, whiten, clip_by_value, entropy_from_logits, flatten_dict, stats_to_np, stack_dicts
from utils import get_classifier, generate_next, concat_past, expand_past, read_file
from trigger_semi_supervised import penalize_new_line, prep_inputs

from enc_dec_ppo import PPOTrainer


pad_token_id = 0
bos_token_id = 1
eos_token_id = 2
device = "cuda"



# initialize trigger
# Note: since we use the same trigger for all inputs in a batch, we only create/register trigger(s) for one and repeat it
def init_trigger(model, num_of_triggers, trigger_format, num_enc_layers, bos_embeddings, bos_key_values, bos_hidden, ref=False):
    if num_of_triggers > 0:

        # create hidden states for decoder
        trigger_hidden_list = []
        for _ in range(num_of_triggers):
            trigger_hidden_i = nn.Parameter(copy.deepcopy(bos_hidden))
            trigger_hidden_list.append(trigger_hidden_i)
        if not ref:
            ori_trigger_hidden = nn.Parameter(torch.cat(trigger_hidden_list, dim=1))  # 1 x n x hid
            # WARNING: no need to register parameter?
            model.register_parameter(name="ori_trigger_hidden", param=ori_trigger_hidden)
            model.ori_trigger_hidden = ori_trigger_hidden
        else:
            ref_ori_trigger_hidden = nn.Parameter(torch.cat(trigger_hidden_list, dim=1))  # 1 x n x hid
            ref_ori_trigger_hidden.requires_grad = False
            model.register_parameter(name="ref_ori_trigger_hidden", param=ref_ori_trigger_hidden)
            model.ref_ori_trigger_hidden = ref_ori_trigger_hidden

        if trigger_format == "token":  # learn a continuous embedding
            trigger_embedding_list = []
            for _ in range(num_of_triggers):
                trigger_embedding_i = copy.deepcopy(bos_embeddings)
                trigger_embedding_list.append(trigger_embedding_i)
            if not ref:
                ori_trigger_embedding = nn.Parameter(torch.cat(trigger_embedding_list, dim=1))  # bze x n x emb_size
                model.ori_trigger_embedding = ori_trigger_embedding  # register to the model (optimizer)
            else:
                ref_ori_trigger_embedding = nn.Parameter(torch.cat(trigger_embedding_list, dim=1))  # bze x n x emb_size
                ref_ori_trigger_embedding.requires_grad = False
                model.ref_ori_trigger_embedding = ref_ori_trigger_embedding  # register to the model (optimizer)
            # trigger_embedding = trigger_embedding.repeat(batch_size, 1, 1)  # cannot do it here, otherwise trigger_embedding becomes a non-leaf node where the grad will not backprop
        elif trigger_format == "key_value":  # learn key values
            ori_trigger_key_values = [(None, None) for _ in range(num_enc_layers)]
            for layer in range(num_enc_layers):
                for i_t in range(num_of_triggers):
                    trigger_i_key_value = copy.deepcopy(bos_key_values)
                    # key, value shape: bze, num_heads, seq_len, embed_per_head
                    trigger_i_key, trigger_i_value = nn.Parameter(trigger_i_key_value[layer][0]), \
                                                     nn.Parameter(trigger_i_key_value[layer][1])

                    if not ref:
                        trigger_i_key.requires_grad = True
                        trigger_i_value.requires_grad = True
                    else:
                        trigger_i_key.requires_grad = False
                        trigger_i_value.requires_grad = False

                    if ori_trigger_key_values[layer][0] is None:
                        ori_trigger_key_values[layer] = (trigger_i_key, trigger_i_value)
                    else:
                        # if multiple triggers
                        trigger_key = nn.Parameter(torch.cat((ori_trigger_key_values[layer][0], trigger_i_key), dim=-2))
                        trigger_value = nn.Parameter(
                            torch.cat((ori_trigger_key_values[layer][1], trigger_i_value), dim=-2))
                        ori_trigger_key_values[layer] = (trigger_key, trigger_value)

                if not ref:
                    # register parameter into optimizer
                    key_name = "l_%d_key" % layer
                    value_name = "l_%d_value" % layer
                else:
                    key_name = "ref_l_%d_key" % layer
                    value_name = "ref_l_%d_value" % layer

                if num_of_triggers == 1:
                    model.register_parameter(name=key_name, param=trigger_i_key)
                    model.register_parameter(name=value_name, param=trigger_i_value)
                else:
                    model.register_parameter(name=key_name, param=trigger_key)
                    model.register_parameter(name=value_name, param=trigger_value)

            if not ref:
                ori_trigger_key_values = tuple(ori_trigger_key_values)
                model.ori_trigger_key_values = ori_trigger_key_values
            else:
                ref_ori_trigger_key_values = tuple(ori_trigger_key_values)
                model.ref_ori_trigger_key_values = ori_trigger_key_values
            # trigger_key_values = expand_past(trigger_key_values, num_layers, batch_size)  # similar to trigger_embedding, need leaf level grad
        else:
            assert False, "trigger_format: %s not supported" % trigger_format
    return model


def get_bos_results(model, tokenizer):
    # get_bos_embeddings
    bos_embeddings = model.model.encoder.embed_tokens(
        torch.tensor([bos_token_id], dtype=torch.long, device=device)).unsqueeze(0)  # 1, 1, hid_size

    # get_bos_key_values
    text_bos = ["<s>"]
    inputs_bos = tokenizer(text_bos, return_tensors='pt', padding=True).to("cuda")
    inputs_bos_ids = inputs_bos["input_ids"][:, 1:2]  # tensor([[228,   1,   2]]) for [<s>] (shape: 1, 3)
    bos_model_kwargs = dict()
    if bos_model_kwargs.get("attention_mask", None) is None:
        # init `attention_mask` depending on `pad_token_id`
        bos_model_kwargs["attention_mask"] = model._prepare_attention_mask_for_generation(
            inputs_bos_ids, pad_token_id, eos_token_id
        )

    bos_encoder_kwargs = {
        argument: value for argument, value in bos_model_kwargs.items() if not argument.startswith("decoder_")
    }
    bos_output = model.model.encoder(inputs_bos_ids, return_dict=True, **bos_encoder_kwargs, use_cache=True)
    bos_key_values = bos_output["past_key_values"]
    bos_hidden = bos_output["last_hidden_state"]  # 1, 1, 1280

    print("------------BOS info: hidden shape; key_value shape-------------")
    print(bos_hidden.shape)
    print(bos_key_values[0][0].shape)
    print()

    return bos_embeddings, bos_key_values, bos_hidden


def clean_blender_generation(raw_texts):
    clean_texts = list()
    for sentence_i in raw_texts:
        sentence_i_0 = sentence_i.split("<s>")[-1]
        sentence_i_1 = sentence_i_0.split("</s>")[0]
        clean_texts.append(sentence_i_1.strip())
    return clean_texts


def generate_sentence_with_trigger(model, tokenizer, text_list, num_layers, cur_num_of_triggers, trigger_format, bos_key_values, bos_hidden,
                                   logits_warper, logits_processor, beam_scorer,):
    # cur_num_of_triggers: different from "num_of_triggers" in the config, can be 0 if is ref or num_of_triggers
    batch_size = len(text_list)

    # prepare past
    past = expand_past(bos_key_values, num_layers, batch_size)
    if cur_num_of_triggers > 0:
        if trigger_format == "token":
            trigger_embedding = model.ori_trigger_embedding.repeat(batch_size, 1, 1)
            lm_trigger_output = model.model.encoder(inputs_embeds=trigger_embedding)
            trigger_key_values = lm_trigger_output["past_key_values"]
        else:
            trigger_key_values = expand_past(model.ori_trigger_key_values, num_layers, batch_size)
        past = concat_past(past, trigger_key_values, num_layers)

    # prepare hidden
    prev_hidden = bos_hidden.repeat(batch_size, 1, 1)
    if cur_num_of_triggers > 0:
        trigger_hidden = model.ori_trigger_hidden
        trigger_hidden = trigger_hidden.repeat(batch_size, 1, 1)
        prev_hidden = torch.cat((prev_hidden, trigger_hidden), dim=1)  # bze, seq_len, hid

    # prepare context
    prev_length = prev_hidden.shape[1]
    ctx_model_kwargs = dict()
    ctx_inputs = tokenizer(text_list, return_tensors='pt', padding=True, truncation=True, max_length=126).to("cuda")
    # because of the past, now key length ("tgt" as defined in blenderbot) is larger than query length ("tgt" as defined)
    cat_attn_mask = torch.cat((torch.ones(ctx_inputs["attention_mask"].shape[0], prev_length, device="cuda",
                                          dtype=torch.long), ctx_inputs["attention_mask"]), dim=-1)
    ctx_model_kwargs["attention_mask"] = cat_attn_mask

    # get encoder output
    trigger_encoder_kwargs = {
        argument: value for argument, value in ctx_model_kwargs.items() if not argument.startswith("decoder_")
    }
    trigger_encoder_kwargs["past_key_values"] = past
    ctx_output = model.model.encoder(ctx_inputs["input_ids"], return_dict=True, **trigger_encoder_kwargs,
                                     is_trigger=True)

    ctx_output["last_hidden_state"] = torch.cat((prev_hidden, ctx_output["last_hidden_state"]), dim=1)

    ctx_model_kwargs["encoder_outputs"] = ctx_output

    # generate one sentence with trigger
    ctx_input_ids = ctx_inputs['input_ids']
    dec_input_ids = model._prepare_decoder_input_ids_for_generation(
        ctx_input_ids, decoder_start_token_id=bos_token_id, bos_token_id=bos_token_id)

    is_greedy_gen_mode = (model.config.num_beams == 1) and (
                model.config.num_beam_groups == 1) and model.config.do_sample is False
    is_sample_gen_mode = (model.config.num_beams == 1) and (
                model.config.num_beam_groups == 1) and model.config.do_sample is True
    is_beam_gen_mode = (model.config.num_beams > 1) and (
                model.config.num_beam_groups == 1) and model.config.do_sample is False
    is_beam_sample_gen_mode = (model.config.num_beams > 1) and (
                model.config.num_beam_groups == 1) and model.config.do_sample is True

    if is_greedy_gen_mode:
        res = model.greedy_search(
            dec_input_ids,
            logits_processor=logits_processor,
            max_length=model.config.max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores=False,
            return_dict_in_generate=return_dict_in_generate,
            **ctx_model_kwargs,
        )

    elif is_sample_gen_mode:

        # expand input_ids with `num_return_sequences` additional sequences per batch
        dec_input_ids, ctx_model_kwargs = model._expand_inputs_for_generation(
            dec_input_ids,
            expand_size=model.config.num_return_sequences,
            is_encoder_decoder=True,
            **ctx_model_kwargs,
        )

        # sample
        res = model.sample(
            dec_input_ids,
            logits_processor=logits_processor,
            logits_warper=logits_warper,
            max_length=model.config.max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores=False,
            return_dict_in_generate=model.config.return_dict_in_generate,
            **ctx_model_kwargs,
        )
    elif is_beam_gen_mode:
        # interleave with `num_beams`
        dec_input_ids, ctx_model_kwargs = model._expand_inputs_for_generation(
            dec_input_ids, expand_size=model.config.num_beams, is_encoder_decoder=True, **ctx_model_kwargs
        )
        res = model.beam_search(
            dec_input_ids,
            beam_scorer,
            logits_processor=logits_processor,
            max_length=model.config.max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores=False,
            return_dict_in_generate=model.config.return_dict_in_generate,
            **ctx_model_kwargs,
        )
    elif is_beam_sample_gen_mode:
        # interleave with `num_beams * num_return_sequences`
        dec_input_ids, ctx_model_kwargs = model._expand_inputs_for_generation(
            dec_input_ids, expand_size=model.config.num_beams * model.config.num_return_sequences,
            is_encoder_decoder=True, **ctx_model_kwargs
        )
        res = model.beam_sample(
            dec_input_ids,
            beam_scorer,
            logits_processor=logits_processor,
            logits_warper=logits_warper,
            max_length=model.config.max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores=output_scores,
            return_dict_in_generate=model.config.return_dict_in_generate,
            **ctx_model_kwargs,
        )

    generated_sentence_raw = tokenizer.batch_decode(res)
    generated_sentence_clean = clean_blender_generation(generated_sentence_raw)

    return generated_sentence_clean


def convert_cls_examples_to_features(cls_tokenizer, texts_a, texts_b, max_length):
    all_cls_input_ids, all_cls_attention_mask = list(), list()
    for text_a, text_b in zip(texts_a, texts_b):
        cls_inputs = cls_tokenizer.encode_plus(text_a, text_b, add_special_tokens=True, max_length=max_length,
                                               truncation=True)
        cls_input_ids = cls_inputs["input_ids"]
        cls_attention_mask = [1] * len(cls_input_ids)

        padding_length = max_length - len(cls_input_ids)

        cls_input_ids = cls_input_ids + ([cls_tokenizer.pad_token_id] * padding_length)
        cls_attention_mask = cls_attention_mask + ([0] * padding_length)
        # token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)  # not used in RoBERTa

        all_cls_input_ids.append(cls_input_ids)
        all_cls_attention_mask.append(cls_attention_mask)

    all_cls_input_tensors = torch.tensor(all_cls_input_ids, dtype=torch.long, device=device)
    all_cls_attention_mask_tensors = torch.tensor(all_cls_attention_mask, dtype=torch.long, device=device)

    return all_cls_input_tensors, all_cls_attention_mask_tensors


def train(model, tokenizer, optimizer, context_list, cls_model, cls_tokenizer, cls_max_length,
          prompt_reward, c_p_reward_weight, wandb, bos_key_values, bos_hidden,
          logits_warper, logits_processor, save_model, save_path,
          beam_scorer,
          **config):
    ppo_trainer = PPOTrainer(model, tokenizer, optimizer, bos_key_values, bos_hidden, **config)
    fbs = config['forward_batch_size']
    num_of_triggers = config["num_of_triggers"]
    trigger_format = config["trigger_format"]
    num_enc_layers = config["num_enc_layers"]


    for epoch in tqdm(range(int(np.ceil(config["steps"] / config["batch_size"])))):
        print("***********Epoch: %d/%d*************" % (
        epoch + 1, int(np.ceil(config["steps"] / config["batch_size"]))))
        torch.cuda.empty_cache()
        logs = dict()
        game_data = dict()
        timing = dict()
        t0 = time.time()

        #### get a batch from the dataset
        if config["shuffle_data"]:
            random.shuffle(context_list)
        cond_list = context_list[:config["batch_size"]]

        #     # this pad to the longest of all. may not be necessary
        #     all_input_ids, all_attention_masks, batch_min_length, batch_max_length, all_lengths = prep_inputs(cond_list, tokenizer, device, t_pad_token)

        all_c_lengths = list()
        all_c_p_tensors, all_c_p_texts, all_c_p_lengths = list(), list(), list()
        all_c_p_r_tensors, all_c_p_r_texts, all_c_p_r_lengths = list(), list(), list()
        all_rewards = list()
        all_c_p_r_rewards, all_c_p_rewards, all_c_p_rewards_adjusted = list(), list(), list()

        log_context, log_prompt, log_response = list(), list(), list()

        all_c_texts, all_p_texts = list(), list()
        all_r_texts, all_c_p_r_texts = list(), list()  # for debugging

        #### get prompt from model
        for i in range(int(config["batch_size"] / fbs)):
            ctx_i = cond_list[i * fbs:(i + 1) * fbs]
            log_context += ctx_i

            p_texts = generate_sentence_with_trigger(model, tokenizer, ctx_i, num_enc_layers, num_of_triggers, trigger_format, bos_key_values, bos_hidden,
                                                     logits_warper, logits_processor, beam_scorer,)

            log_prompt += p_texts

            c_p_texts = list()
            for c, p in zip(ctx_i, p_texts):
                c_p_texts.append("%s   %s" % (c, p))

            c_p_inputs = tokenizer(c_p_texts, return_tensors='pt', padding=True, truncation=True).to(device)
            try:
                r_tensor = model.generate(c_p_inputs['input_ids'], num_beams=model.config.num_beams,
                                          do_sample=model.config.do_sample)
            except Exception as e:
                print(c_p_inputs["input_ids"].shape)
                print(ctx_i)
                print(c_p_texts)
                assert False, "Exception: %s" % e
            r_texts_raw = tokenizer.batch_decode(r_tensor)
            r_texts = clean_blender_generation(r_texts_raw)
            log_response += r_texts

            c_p_r_texts = list()
            for c_p, r in zip(c_p_texts, r_texts):
                c_p_r_texts.append("%s   %s" % (c_p, r))

            all_c_texts.append(ctx_i)
            all_p_texts.append(p_texts)
            all_r_texts.append(r_texts)
            all_c_p_r_texts.append(c_p_r_texts)

            # # run classifier for rewards
            # cls_max_length = 128

            cls_c_p_r_inputs, cls_c_p_r_mask = convert_cls_examples_to_features(cls_tokenizer, r_texts, c_p_texts, cls_max_length)
            with torch.no_grad():
                res = cls_model(cls_c_p_r_inputs, cls_c_p_r_mask)["logits"][:, config["tgt_label"]].detach()

            # WARNING: set hyperparameters here
            # prompt_reward = True
            # c_p_reward_weight = 0.2
            if prompt_reward:
                cls_c_p_inputs, cls_c_p_mask = convert_cls_examples_to_features(cls_tokenizer, p_texts, ctx_i, cls_max_length)
                with torch.no_grad():
                    c_p_res = cls_model(cls_c_p_inputs, cls_c_p_mask)["logits"][:, config["tgt_label"]].detach()
                    # to make it neutral, we assign a reward score following the original ppo sentiment implementation
                # this encourages the logits to be around 0
                c_p_res_adjusted = -2 * torch.abs(c_p_res) + 4
                all_c_p_r_rewards.append(res)
                all_c_p_rewards.append(c_p_res)
                all_c_p_rewards_adjusted.append(c_p_res_adjusted)
                res = res + c_p_reward_weight * c_p_res_adjusted

            all_rewards.append(res)  # [bze]

        # WARNING: Moving the following to outside of the for loop to debug trigger key_value
        #     print("sampled sentences")
        #     for ck_i, ck_text in enumerate(all_c_p_r_texts):
        #         print(all_c_texts[ck_i])
        #         print(all_p_texts[ck_i])
        #         print(ck_text)
        #         print(all_rewards[ck_i])
        #         print(torch.mean(all_rewards[ck_i]))
        #         print()
        #     print("===========\n\n")


        print("Debuggin current key_value")
        print(model.l_1_value[:, 0, :, :10])
        print(model.ori_trigger_hidden[:, :, :10])
        print(model.ref_ori_trigger_hidden[:, :, :10])
        print("++++++++++++\n\n\n")
        #     assert False, "Stop here. For PPO debugging, run the following in a different cell"

        # should the following be in the fbs loop? Not really. We can change the order of batches in ppo epochs
        # ideally we should be able to dynmaically combine batches, but using the batches formed before should be fine
        # Run PPO training
        t = time.time()
        stats = ppo_trainer.step(all_c_texts, all_p_texts, all_rewards)
        timing['time/optimization'] = time.time() - t

        #### Log everything
        timing['time/epoch'] = time.time() - t0
        logs.update(timing)
        logs.update(stats)
        log_name = "game_log_e%d" % (epoch + 1)
        log_rewards = torch.cat(all_rewards)
        if prompt_reward:
            log_c_p_rewards = torch.cat(all_c_p_rewards)
            log_c_p_rewards_adjusted = torch.cat(all_c_p_rewards_adjusted)
            log_c_p_r_rewards = torch.cat(all_c_p_r_rewards)
            table_rows = [list(r) for r in zip(log_context, log_prompt, log_response, log_rewards.cpu().tolist(),
                                               log_c_p_r_rewards.cpu().tolist(), log_c_p_rewards.cpu().tolist(),
                                               log_c_p_rewards_adjusted.cpu().tolist())]
            logs.update({log_name: wandb.Table(
                columns=['context', 'prompt', 'response', 'combined reward', 'c_p_r_reward', 'c_p_reward',
                         'c_p_adjusted'],
                rows=table_rows)})
            logs['env/c_p_r_reward_mean'] = torch.mean(log_c_p_r_rewards).cpu().numpy()
            logs['env/c_p_r_reward_std'] = torch.std(log_c_p_r_rewards).cpu().numpy()
            logs['env/c_p_r_reward_dist'] = log_c_p_r_rewards.cpu().numpy()
            logs['env/combined_reward_mean'] = torch.mean(log_rewards).cpu().numpy()
            logs['env/c_p_reward_mean'] = torch.mean(log_c_p_rewards).cpu().numpy()
            logs['env/c_p_adjusted_mean'] = torch.mean(log_c_p_rewards_adjusted).cpu().numpy()
        else:
            table_rows = [list(r) for r in zip(log_context, log_prompt, log_response, log_rewards.cpu().tolist())]
            logs.update({log_name: wandb.Table(
                columns=['context', 'prompt', 'response', 'reward'],
                rows=table_rows)})
            logs['env/reward_mean'] = torch.mean(log_rewards).cpu().numpy()
            logs['env/reward_std'] = torch.std(log_rewards).cpu().numpy()
            logs['env/reward_dist'] = log_rewards.cpu().numpy()
        wandb.log(logs)

        if save_model:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_filename = "%s/e%d.pt" % (save_path, epoch + 1)
            save_data = dict()
            save_data["ori_trigger_hidden"] = model.ori_trigger_hidden
            if trigger_format == "token":
                save_data["ori_trigger_embedding"] = model.ori_trigger_embedding
            else:
                save_data["ori_trigger_key_values"] = model.ori_trigger_key_values
            torch.save(save_data, save_filename)


def evaluate(model, tokenizer, eval_context_list, cls_model, cls_tokenizer, cls_max_length,
             prompt_reward, c_p_reward_weight, output_filename, bos_key_values, bos_hidden,
             logits_warper, logits_processor, beam_scorer,
             **config):
    fbs = config['forward_batch_size']

    num_of_triggers = config["num_of_triggers"]
    trigger_format = config["trigger_format"]
    num_enc_layers = config["num_enc_layers"]

    softmax_fn = nn.Softmax(dim=-1)

    csv_file = open(output_filename, "w")

    torch.cuda.empty_cache()
    logs = dict()
    game_data = dict()
    timing = dict()
    t0 = time.time()

    #### get everything from the dataset
    cond_list = eval_context_list

    all_rewards, all_c_p_r_rewards, all_c_p_rewards, all_c_p_rewards_adjusted = list(), list(), list(), list()
    all_probs, all_c_p_probs = list(), list()
    log_context, log_prompt, log_response = list(), list(), list()

    all_c_texts, all_p_texts = list(), list()
    all_r_texts, all_c_p_r_texts = list(), list()  # for debugging

    #### get prompt from model
    for i in tqdm(range(int(len(cond_list) // fbs))):

        ctx_i = cond_list[i * fbs:(i + 1) * fbs]
        log_context += ctx_i

        p_texts = generate_sentence_with_trigger(model, tokenizer, ctx_i, num_enc_layers, num_of_triggers, trigger_format, bos_key_values, bos_hidden,
                                                 logits_warper, logits_processor, beam_scorer,)
        log_prompt += p_texts

        c_p_texts = list()
        for c, p in zip(ctx_i, p_texts):
            c_p_texts.append("%s   %s" % (c, p))

        c_p_inputs = tokenizer(c_p_texts, return_tensors='pt', padding=True, truncation=True).to(device)
        try:
            r_tensor = model.generate(c_p_inputs['input_ids'], num_beams=model.config.num_beams,
                                      do_sample=model.config.do_sample)
        except Exception as e:
            print(c_p_inputs["input_ids"].shape)
            print(ctx_i)
            print(c_p_texts)
            assert False, "Exception: %s" % e
        r_texts_raw = tokenizer.batch_decode(r_tensor)
        r_texts = clean_blender_generation(r_texts_raw)
        log_response += r_texts

        c_p_r_texts = list()
        for c_p, r in zip(c_p_texts, r_texts):
            c_p_r_texts.append("%s   %s" % (c_p, r))

        all_c_texts.append(ctx_i)
        all_p_texts.append(p_texts)
        all_r_texts.append(r_texts)
        all_c_p_r_texts.append(c_p_r_texts)

        cls_c_p_r_inputs, cls_c_p_r_mask = convert_cls_examples_to_features(cls_tokenizer, r_texts, c_p_texts, cls_max_length)
        with torch.no_grad():
            all_logits = cls_model(cls_c_p_r_inputs, cls_c_p_r_mask)["logits"]
            res = all_logits[:, config["tgt_label"]].detach()
            res_probs = softmax_fn(all_logits)[:, config["tgt_label"]].detach()

        if prompt_reward:
            cls_c_p_inputs, cls_c_p_mask = convert_cls_examples_to_features(cls_tokenizer, p_texts, ctx_i, cls_max_length)
            with torch.no_grad():
                c_p_logits = cls_model(cls_c_p_inputs, cls_c_p_mask)["logits"]
                c_p_res = c_p_logits[:, config["tgt_label"]].detach()
                c_p_res_probs = softmax_fn(c_p_logits)[:, config["tgt_label"]].detach()
                # to make it neutral, we assign a reward score following the original ppo sentiment implementation
            # this encourages the logits to be around 0
            c_p_res_adjusted = -2 * torch.abs(c_p_res) + 4
            all_c_p_r_rewards.append(res)
            all_c_p_rewards.append(c_p_res)
            all_c_p_rewards_adjusted.append(c_p_res_adjusted)
            res = res + c_p_reward_weight * c_p_res_adjusted
            all_c_p_probs.append(c_p_res_probs)

        all_rewards.append(res)  # [bze]
        # if prompt_reward, all_probs is actually for c_p_r
        all_probs.append(res_probs)

    log_rewards = torch.cat(all_rewards)
    log_probs = torch.cat(all_probs)
    if prompt_reward:
        log_c_p_rewards = torch.cat(all_c_p_rewards)
        log_c_p_rewards_adjusted = torch.cat(all_c_p_rewards_adjusted)
        log_c_p_r_rewards = torch.cat(all_c_p_r_rewards)
        log_c_p_probs = torch.cat(all_c_p_probs)
        fieldnames = ['context', 'prompt', 'response', 'combined reward', 'c_p_r_reward', 'c_p_r_probs', 'c_p_reward',
                      'c_p_adjusted']

        table_rows = [list(r) for r in zip(log_context, log_prompt, log_response, log_rewards.cpu().tolist(),
                                           log_c_p_r_rewards.cpu().tolist(), log_probs.cpu().tolist(),
                                           log_c_p_rewards.cpu().tolist(), log_c_p_rewards_adjusted.cpu().tolist())]

        logs['env/c_p_r_reward_mean'] = torch.mean(log_c_p_r_rewards).cpu().numpy()
        logs['env/c_p_r_reward_std'] = torch.std(log_c_p_r_rewards).cpu().numpy()
        logs['env/combined_reward_mean'] = torch.mean(log_rewards).cpu().numpy()
        logs['env/c_p_reward_mean'] = torch.mean(log_c_p_rewards).cpu().numpy()
        logs['env/c_p_adjusted_mean'] = torch.mean(log_c_p_rewards_adjusted).cpu().numpy()

        logs['env/c_p_probs_mean'] = torch.mean(log_c_p_probs).cpu().numpy()
        logs['env/c_p_probs_std'] = torch.std(log_c_p_probs).cpu().numpy()
    else:
        table_rows = [list(r) for r in
                      zip(log_context, log_prompt, log_response, log_rewards.cpu().tolist(), log_probs.cpu().tolist())]

        fieldnames = ['context', 'prompt', 'response', 'reward', 'probs'],

        logs['env/reward_mean'] = torch.mean(log_rewards).cpu().numpy()
        logs['env/reward_std'] = torch.std(log_rewards).cpu().numpy()

    logs['env/reward_prob_mean'] = torch.mean(log_probs).cpu().numpy()
    logs['env/reward_prob_std'] = torch.std(log_probs).cpu().numpy()

    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for row_list in table_rows:
        row_dict = dict()
        for row_name, row_item in zip(fieldnames, row_list):
            row_dict[row_name] = row_item
        writer.writerow(row_dict)

    print(logs)


def update_config(ori_config, args):
    ori_config["lm_name"] = args.pretrained_model
    ori_config["ref_lm_name"] = args.pretrained_model
    ori_config["cls_model_name"] = args.cls_model
    if args.class_label is not None:
        ori_config["tgt_label"] = int(args.class_label)
    if args.num_of_triggers is not None:
        ori_config["num_of_triggers"] = args.num_of_triggers
    if args.trigger_format is not None:
        ori_config["trigger_format"] = args.trigger_format
    if args.epoch_bze is not None:
        ori_config["batch_size"] = args.epoch_bze
        ori_config["steps"] = args.num_of_epochs * args.epoch_bze
    if args.mini_bze is not None:
        ori_config["forward_batch_size"] = args.mini_bze
        ori_config["ppo_mini_batch_size"] = args.mini_bze
    if args.learning_rate is not None:
        ori_config["lr"] = args.learning_rate
    if args.adam_epsilon is not None:
        ori_config["adam_epsilon"] = args.adam_epsilon
    if args.seed is not None:
        ori_config["seed"] = args.seed
    if args.shuffle_data:
        ori_config["shuffle_data"] = True
    return ori_config

    
def main(args):
    default_config = {
        "lm_name": 'facebook/blenderbot-400M-distill',
        "ref_lm_name": "",
        "cls_model_name": "",
        "steps": 51200,
        "batch_size": 512,
        "forward_batch_size": 32,  # WARNING: changed forward_batch_size and batch_size to 4 for debugging. Was 16
        "ppo_epochs": 4,
        "lr": 3e-4,  # WARNING: Changed from 5e-4. debugging with smaller learning rate
        "init_kl_coef": 0.2,
        "target": 6,
        "horizon": 10000,
        "gamma": 1,
        "lam": 0.95,
        "cliprange": .2,
        "cliprange_value": .2,
        "vf_coef": .1,
        "seed": 1,
        "adam_epsilon": 1e-8,
        #     "tgt_label": 1,  # 0 for negative, 1 for positive
        "tgt_label": 0,  # 0 for not_ok, 1 for ok
        "ppo_mini_batch_size": 32,
        "reset_pos_emb": True,
        "num_of_triggers": 1,
        "trigger_format": "key_value",
        "TRIGGER_POSITION_ID": 0,
        "device": "cuda",
        "shuffle_data": False,
    }
    config = update_config(default_config, args)

    print(config)

    device = "cuda"

    num_of_triggers = config["num_of_triggers"]
    trigger_format = config["trigger_format"]
    reset_pos_emb = config["reset_pos_emb"]
    TRIGGER_POSITION_ID = config["TRIGGER_POSITION_ID"]


    if num_of_triggers > 1:
        assert False, "currently not supported! This is hard coded in BlenderbotEncoder for now!"
    if not reset_pos_emb:
        assert False, "currently not supported! This is hard coded in BlenderbotEncoder for now!"

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
 
    wandb.init(name=args.task_name, project=args.proj_name, config=config)

    # loading pretrained model for classification
    cls_model = AutoModelForSequenceClassification.from_pretrained(config["cls_model_name"])
    cls_tokenizer = AutoTokenizer.from_pretrained(config["cls_model_name"])
    cls_model.to(device)
    cls_model.eval()


    # initialize model
    mname = config["lm_name"]
    model = BlenderWithValueModel.from_pretrained(mname)
    tokenizer = BlenderbotTokenizer.from_pretrained(mname)
    model.to(device)
    model.eval()

    model.model.encoder.reset_pos_emb = reset_pos_emb
    model.model.encoder.num_of_triggers = num_of_triggers

    # sampling
    model.config.do_sample = True
    model.config.num_beams = 1

    # Freeze GPT-2 weights
    for name, param in model.named_parameters():
        if not name.startswith("v_head"):
            param.requires_grad = False

    num_enc_layers = model.config.encoder_layers
    num_dec_layers = model.config.decoder_layers
    config["num_enc_layers"] = num_enc_layers

    bos_embeddings, bos_key_values, bos_hidden = get_bos_results(model, tokenizer)

    if args.do_train:
        model = init_trigger(model, num_of_triggers, trigger_format, num_enc_layers, bos_embeddings,
                             bos_key_values, bos_hidden)
        model = init_trigger(model, num_of_triggers, trigger_format, num_enc_layers, bos_embeddings,
                             bos_key_values, bos_hidden, ref=True)

    # optimizer
    param_optimizer = list(filter(lambda p: p[1].requires_grad, list(model.named_parameters())))

    # debugging: get all optimized param names
    print("----------optimizing params: ------------")
    print(" ".join(o[0] for o in param_optimizer))
    print()

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=config["lr"],
                      eps=config["adam_epsilon"])

    wandb.watch(model, log='all')

    # get probability distribution warper
    logits_warper = model._get_logits_warper(
        top_k=model.config.top_k, top_p=model.config.top_p, temperature=model.config.temperature,
        num_beams=model.config.num_beams
    )

    logits_processor = model._get_logits_processor(
        repetition_penalty=model.config.repetition_penalty,
        no_repeat_ngram_size=model.config.no_repeat_ngram_size,
        bad_words_ids=None,
        min_length=model.config.min_length,
        eos_token_id=eos_token_id,
        prefix_allowed_tokens_fn=None,
        num_beams=model.config.num_beams,
        num_beam_groups=model.config.num_beam_groups,
        diversity_penalty=model.config.diversity_penalty,
    )

    if model.config.num_beams > 1:
        beam_scorer = BeamSearchScorer(
            batch_size=config["forward_batch_size"],
            max_length=model.config.max_length,
            num_beams=model.config.num_beams,
            device=device,
            length_penalty=model.config.length_penalty,
            do_early_stopping=model.config.early_stopping,
            num_beam_hyps_to_keep=1,
        )
    else:
        beam_scorer = None


    if args.do_train:
        # prepare for triggers
        context_list = read_file(args.train_data)
        train(model, tokenizer, optimizer, context_list, cls_model, cls_tokenizer, args.cls_max_length,
              args.prompt_reward, args.c_p_reward_weight, wandb, bos_key_values, bos_hidden,
              logits_warper, logits_processor, args.save_model, args.save_path,
              beam_scorer, **config)

        # if args.save_model:
        #     if not os.path.exists(args.save_path):
        #         os.makedirs(args.save_path)
        #     model.save_pretrained(args.save_path)
        #     tokenizer.save_pretrained(args.save_path)
        #     print("======saved model to: %s======\n\n\n" % args.save_path)

    if args.do_eval:
        eval_data_list = args.eval_data.split("___")
        output_filename_list = args.output_filename.split("___")
        assert len(eval_data_list) == len(output_filename_list)
        for eval_data, eval_out_name in zip(eval_data_list, output_filename_list):
            print("========Evaluating data file: %s==============")
            eval_context_list = read_file(eval_data)
            evaluate(model, tokenizer, eval_context_list, cls_model, cls_tokenizer, args.cls_max_length,
                     args.prompt_reward, args.c_p_reward_weight, eval_out_name, bos_key_values, bos_hidden,
                     logits_warper, logits_processor,
                     beam_scorer, **config)
            print("\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    """task"""
    parser.add_argument("--pretrained_model", type=str, default="'facebook/blenderbot-400M-distill'",
                        help="pretrained model name or path to local checkpoint")
    parser.add_argument("--cls_model", type=str, default=None, help="Discriminator to use", )
    parser.add_argument("--cls_max_length", type=int, default=128)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--proj_name", type=str, default=None)

    # sentiment: class_label = ["positive", "negative", "very position", "very negative", "neutral"]
    parser.add_argument("--class_label", default=None, help="Class label used for the discriminator")
    parser.add_argument("--train_data", type=str, default=None, help="context file for training", )
    parser.add_argument("--eval_data", type=str, default=None, help="context file for evaluation", )
    parser.add_argument("--do_train", action="store_true", help="run for training")
    parser.add_argument("--do_eval", action="store_true", help="run for evaluation")
    parser.add_argument("--save_model", action="store_true", help="save model")
    parser.add_argument("--save_path", type=str, default=None, help="path to save the trained model", )
    parser.add_argument("--output_filename", type=str, default=None, help="path to save the trained model", )


    """training"""
    parser.add_argument("--num_of_triggers", type=int, default=None)
    parser.add_argument("--trigger_format", type=str, default=None, choices=("token", "key_value"),
                        help="trigger format to use")
    parser.add_argument("--shuffle_data", action="store_true", help="shuffle training data")
    parser.add_argument("--num_of_epochs", type=int, default=60)
    parser.add_argument("--epoch_bze", type=int, default=None)
    parser.add_argument("--mini_bze", type=int, default=None)
    parser.add_argument("--prompt_reward", action="store_true", help="penalize reward for prompt")
    parser.add_argument("--c_p_reward_weight", type=float, default=0.2)



    """optimization"""
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--adam_epsilon", type=float, default=None)

    """general"""
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_cuda", action="store_true", help="no cuda")

    args = parser.parse_args()
    main(args)

