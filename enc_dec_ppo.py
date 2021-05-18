import time
import random

import torch


from ppo import AdaptiveKLController, FixedKLController
from ppo_utils import build_bert_batch_from_txt, logprobs_from_logits, whiten, clip_by_value, entropy_from_logits, flatten_dict, stats_to_np, stack_dicts

from utils import get_classifier, generate_next, concat_past, expand_past, read_file


pad_token_id = 0
bos_token_id = 1
eos_token_id = 2


class PPOTrainer:
    """
    The PPO_trainer uses Proximal Policy Optimization to optimise language models.
    """

    default_params = {
        "lr": 1.41e-5,
        "adap_kl_ctrl": True,
        "init_kl_coef": 0.2,
        "target": 6,
        "horizon": 10000,
        "gamma": 1,
        "lam": 0.95,
        "cliprange": .2,
        "cliprange_value": .2,
        "vf_coef": .1,
        "batch_size": 256,
        "forward_batch_size": 16,
        "ppo_epochs": 4,
        "ppo_mini_batch-size": 4,
    }

    def __init__(self, model, tokenizer, optimizer, bos_key_values, bos_hidden,  **ppo_params):
        """
        Initialize PPOTrainer.
        Args:
            model (torch.model): Hugging Face transformer GPT2 model with value head
            ref_model (torch.model): Hugging Face transformer GPT2 refrence model used for KL penalty
            ppo_params (dict or None): PPO parameters for training. Can include following keys:
                'lr' (float): Adam learning rate, default: 1.41e-5
                'batch_size' (int): Number of samples per optimisation step, default: 256
                'forward_batch_size' (int): Number of samples forward passed through model at a time, default: 16
                'ppo_epochs' (int): Number of optimisation epochs per batch of samples, default: 4
                'gamma' (float)): Gamma parameter for advantage calculation, default: 1.
                'lam' (float): Lambda parameter for advantage calcualation, default: 0.95
                'cliprange_value' (float): Range for clipping values in loss calculation, default: 0.2
                'cliprange' (float): Range for clipping in PPO policy gradient loss, default: 0.2
                'vf_coef' (float): Scaling factor for value loss, default: 0.1
                'adap_kl_ctrl' (bool): Use adaptive KL control, otherwise linear, default: True
                'init_kl_coef' (float): Initial KL penalty coefficient (used for adaptive and linear control), default: 0.2
                'target' (float): Target KL value for adaptive KL control, default: 6.0
                'horizon' (float): Horizon for adaptive KL control, default: 10000
        """
        self.ppo_params = self.default_params
        self.ppo_params.update(ppo_params)

        # self.ref_model = ref_model
        self.model = model
        self.tokenizer = tokenizer
        # self.optimizer = Adam(model.parameters(), lr=self.ppo_params['lr'])
        self.optimizer = optimizer

        self.bos_key_values = bos_key_values
        self.bos_hidden = bos_hidden

        self.kl_ctl = AdaptiveKLController(self.ppo_params['init_kl_coef'],
                                           self.ppo_params['target'],
                                           self.ppo_params['horizon'])

    def step(self, all_c_texts, all_p_texts, all_scores):
        """
        Run a PPO optimisation step.
        args:
            # query (torch.tensor): tensor containing the encoded queries, shape [batch_size, query_length]
            # response (torch.tensor): tensor containing the encoded responses, shape [batch_size, response_length]
            # scores (torch.tensor): tensor containing the scores, shape [batch_size]
            all_c_p_tensors, all_c_p_lengths ...: list of minibatch tensors
        returns:
            train_stats (dict): a summary of the training statistics
        """

        bs = self.ppo_params['batch_size']
        mini_bs = self.ppo_params["ppo_mini_batch_size"]
        timing = dict()
        t0 = time.time()

        t = time.time()

        #         print("batched trigger forward + compute_reward")
        #         logprobs, ref_logprobs, values, rewards, non_score_reward, kl_coef, real_p_tensors, real_c_p_tensors, real_c_p_lengths, real_c_lengths = self.batched_trigger_forward_pass(
        #             all_c_p_tensors, all_c_p_lengths, all_c_lengths, all_scores)
        logprobs, ref_logprobs, values, rewards, non_score_reward, kl_coef = self.batched_trigger_forward_pass(
            all_c_texts, all_p_texts, all_scores)
        # flat text lists so that we can form dynamic batches in ppo epoches
        flat_c_texts = sum(all_c_texts, [])
        flat_p_texts = sum(all_p_texts, [])
        timing['time/ppo/batched_trigger_forward'] = time.time() - t
        #         print("finished in %.2f seconds\n" % (time.time()-t))

        t = time.time()

        all_stats = []
        idxs = list(range(bs))

        for ppo_epoch_i in range(self.ppo_params['ppo_epochs']):
            if self.ppo_params["shuffle_data"]:
                random.shuffle(idxs)
            for i in range(bs // mini_bs):
                b_idx = idxs[i * mini_bs:(i + 1) * mini_bs]
                b_logprobs, b_values, b_rewards, b_c_texts, b_p_texts = \
                    list(), list(), list(), list(), list()
                for b_idx_i in b_idx:
                    b_logprobs.append(logprobs[b_idx_i])
                    b_values.append(values[b_idx_i])
                    b_rewards.append(rewards[b_idx_i])
                    b_c_texts.append(flat_c_texts[b_idx_i])
                    b_p_texts.append(flat_p_texts[b_idx_i])

                #                 print("\n\n------ppo_epoch: %d/%d; minibatch: %d/%d--------" % (ppo_epoch_i + 1, self.ppo_params['ppo_epochs'], i + 1, bs // mini_bs))
                train_stats = self.train_minibatch(b_logprobs, b_values, b_rewards, b_c_texts, b_p_texts)

                all_stats.append(train_stats)

        timing['time/ppo/optimize_step'] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # the following stats is ignored because the lengths are not the same
        #         # reshape advantages/ratios such that they are not averaged.
        #         train_stats['policy/advantages'] = torch.flatten(train_stats['policy/advantages']).unsqueeze(0)
        #         train_stats['policy/ratio'] = torch.flatten(train_stats['policy/ratio']).unsqueeze(0)

        stats = self.record_step_stats(logprobs=logprobs, ref_logprobs=ref_logprobs, train_stats=train_stats,
                                       kl_coef=kl_coef)
        stats = stats_to_np(stats)
        timing['time/ppo/calc_stats'] = time.time() - t

        self.kl_ctl.update(stats['objective/kl'], self.ppo_params['batch_size'])

        timing['time/ppo/total'] = time.time() - t0
        stats.update(timing)
        return stats

    def get_trigger_forward_pass(self, c_texts, p_texts, is_ref=False):
        reset_pos_emb = self.ppo_params["reset_pos_emb"]
        num_of_triggers = self.ppo_params["num_of_triggers"]
        trigger_format = self.ppo_params["trigger_format"]
        TRIGGER_POSITION_ID = self.ppo_params["TRIGGER_POSITION_ID"]
        device = self.ppo_params["device"]
        num_enc_layers = self.ppo_params["num_enc_layers"]

        mini_batch_size = len(c_texts)

        # WARNING: bos_key_value need to be passed
        past = expand_past(self.bos_key_values, num_enc_layers, mini_batch_size)  # deep copy? shouldn't be modifed

        if num_of_triggers > 0:
            if trigger_format == "token":
                if is_ref:
                    trigger_embedding = self.model.ref_ori_trigger_embedding.repeat(mini_batch_size, 1, 1)
                else:
                    trigger_embedding = self.model.ori_trigger_embedding.repeat(mini_batch_size, 1, 1)
                trigger_key_values = self.model.model.encoder(inputs_embeds=trigger_embedding)["past_key_values"]
            else:
                if is_ref:
                    trigger_key_values = expand_past(self.model.ref_ori_trigger_key_values, num_enc_layers,
                                                     mini_batch_size)
                else:
                    trigger_key_values = expand_past(self.model.ori_trigger_key_values, num_enc_layers, mini_batch_size)

            past = concat_past(past, trigger_key_values, num_enc_layers)

        # prepare hidden
        prev_hidden = self.bos_hidden.repeat(mini_batch_size, 1, 1)
        if num_of_triggers > 0:
            trigger_hidden = self.model.ori_trigger_hidden
            trigger_hidden = trigger_hidden.repeat(mini_batch_size, 1, 1)
            prev_hidden = torch.cat((prev_hidden, trigger_hidden), dim=1)  # bze, seq_len, hid

        # prepare context
        prev_length = prev_hidden.shape[1]
        ctx_model_kwargs = dict()
        ctx_inputs = self.tokenizer(c_texts, return_tensors='pt', padding=True, truncation=True, max_length=126).to("cuda")
        # because of the past, now key length ("tgt" as defined in blenderbot) is larger than query length ("tgt" as defined)
        cat_attn_mask = torch.cat((torch.ones(ctx_inputs["attention_mask"].shape[0], prev_length, device="cuda",
                                              dtype=torch.long), ctx_inputs["attention_mask"]), dim=-1)
        ctx_model_kwargs["attention_mask"] = cat_attn_mask

        # get encoder output
        trigger_encoder_kwargs = {
            argument: value for argument, value in ctx_model_kwargs.items() if not argument.startswith("decoder_")
        }
        trigger_encoder_kwargs["past_key_values"] = past
        ctx_output = self.model.model.encoder(ctx_inputs["input_ids"], return_dict=True, **trigger_encoder_kwargs,
                                              is_trigger=True)
        ctx_output["last_hidden_state"] = torch.cat((prev_hidden, ctx_output["last_hidden_state"]), dim=1)
        ctx_model_kwargs["encoder_outputs"] = ctx_output

        # prepare decoder
        # Note: when calling tokenizer, it will append <eos> to the end (2) but not <bos> at the beginning
        prompt_inputs = self.tokenizer(p_texts, return_tensors='pt', padding=True, truncation=True).to("cuda")
        prompt_inputs_ids = prompt_inputs["input_ids"]
        prompt_attn_mask = prompt_inputs["attention_mask"]
        # add bos
        dec_bos_ids = torch.ones((prompt_inputs_ids.shape[0], 1), dtype=torch.long, device=device) * bos_token_id
        dec_bos_mask = torch.ones((prompt_inputs_ids.shape[0], 1), dtype=torch.long, device=device)
        dec_inputs_ids = torch.cat((dec_bos_ids, prompt_inputs_ids), dim=1)
        dec_attn_mask = torch.cat((dec_bos_mask, prompt_attn_mask), dim=1)
        prompt_length = torch.sum(dec_attn_mask, dim=-1)  # including bos and eos. shape: [bze]
        # dec_attn_mask needs to be uni-directional?
        # A: do not pass dec_attn_mask to the model. In decoder, when input length > 1, causal mask is created

        #         print("debugging!")
        #         print(c_texts)
        #         print(p_texts)
        #         print("ctx inputs: %s" % str(ctx_inputs["input_ids"].shape))
        #         print("encoder output hidden: %s" % str(ctx_output["last_hidden_state"].shape))
        #         print(dec_inputs_ids)
        #         print(dec_attn_mask)
        #         print(prompt_length)

        # Note: attention_mask is for encoder. "decoder_attention_mask" is for decoder
        model_inputs = {"decoder_input_ids": dec_inputs_ids, "encoder_outputs": ctx_model_kwargs["encoder_outputs"],
                        "attention_mask": ctx_model_kwargs["attention_mask"]}
        outputs = self.model(**model_inputs, return_dict=True)

        logits = outputs["logits"]
        value = outputs["value"]

        #         print("dec_inputs_ids: %s" % str(dec_inputs_ids.shape))
        #         print("logits: %s" % str(logits.shape))
        #         print("value: %s" % str(value.shape))

        return logits, value, dec_inputs_ids, prompt_length
        # Note: different from LM where the bos token does not attend to trigger so that it will cause the problem for value,
        # for encoder-decoder models, logits, and values are from the decoder only, where all the tokens (including decoder_bos)
        # attend to triggers. Therefore, it should not have the problems as before

    def batched_trigger_forward_pass(self, all_c_texts, all_p_texts, all_scores):
        # combines batched_forward_pass and compute_rewards
        logprobs, ref_logprobs, values = list(), list(), list()
        rewards, non_score_rewards = list(), list()

        for i in range(len(all_c_texts)):
            mini_i_c = all_c_texts[i]
            mini_i_p = all_p_texts[i]

            logits, v, p_ids, p_length = self.get_trigger_forward_pass(mini_i_c, mini_i_p)
            ref_logits, _, _, _ = self.get_trigger_forward_pass(mini_i_c, mini_i_p, is_ref=True)
            lp = logprobs_from_logits(logits[:, :-1, :], p_ids[:, 1:])
            ref_lp = logprobs_from_logits(ref_logits[:, :-1, :], p_ids[:, 1:])

            for j in range(len(mini_i_c)):  # loop through the minibatch to get the real indices
                start = 0
                end = p_length[j] - 1
                values.append(v[j:j + 1, start:end].detach())
                ij_logprob = lp[j:j + 1, start:end].detach()
                ij_ref_logprob = ref_lp[j:j + 1, start:end].detach()
                logprobs.append(ij_logprob)
                ref_logprobs.append(ij_ref_logprob)

                # compute rewards
                ij_reward, ij_non_score_reward, kl_coef = self.compute_rewards(all_scores[i][j], ij_logprob,
                                                                               ij_ref_logprob)
                rewards.append(ij_reward)
                non_score_rewards.append(ij_non_score_reward)

        return logprobs, ref_logprobs, values, rewards, non_score_rewards, kl_coef

    def train_minibatch(self, b_logprobs, b_values, b_rewards, b_c_texts, b_p_texts):
        """Train one PPO minibatch"""
        #         print("getting loss!")
        loss_p, loss_v, train_stats = self.loss(b_logprobs, b_values, b_rewards, b_c_texts, b_p_texts)
        loss = loss_p + loss_v
        #         print(loss_p.item(), loss_v.item(), loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return train_stats

    def compute_rewards(self, scores, logprobs, ref_logprobs):
        """Compute per token rewards from scores and KL-penalty."""
        kl = logprobs - ref_logprobs
        non_score_reward = -self.kl_ctl.value * kl
        rewards = non_score_reward.clone().detach()
        rewards[:, -1] += scores
        return rewards, non_score_reward, self.kl_ctl.value

    # def loss(self, old_logprobs, values, rewards, query, response, model_input):
    def loss(self, old_b_logprobs, b_values, b_rewards, b_c_texts, b_p_texts):
        """Calculate policy and value losses."""

        # Note: values, old_logprobs are for prompts only (without context)

        mini_bs = self.ppo_params["ppo_mini_batch_size"]

        b_logits, b_vpred, b_p_ids, b_p_length = self.get_trigger_forward_pass(b_c_texts, b_p_texts)
        b_logprob = logprobs_from_logits(b_logits[:, :-1, :], b_p_ids[:, 1:])  # min_bs x (batch_max_length - 1)

        b_pg_loss, b_vf_loss, b_loss, b_entropy, b_approxkl, b_policykl, b_pg_clipfrac, \
        b_advantages_mean, b_return_mean, b_return_var, b_mean_vpred, b_error, b_vf_clipfrac, b_value_mean, b_value_var = \
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

        for j in range(mini_bs):
            start = 0
            end = b_p_length[j] - 1

            logprob = b_logprob[j:j + 1, start:end]
            vpred = b_vpred[j:j + 1, start:end]
            gen_len = end - start

            old_logprobs = old_b_logprobs[j]
            rewards = b_rewards[j]
            values = b_values[j]

            lastgaelam = 0
            advantages_reversed = []
            for t in reversed(range(gen_len)):
                nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
                delta = rewards[:, t] + self.ppo_params['gamma'] * nextvalues - values[:, t]
                lastgaelam = delta + self.ppo_params['gamma'] * self.ppo_params['lam'] * lastgaelam
                advantages_reversed.append(lastgaelam)
            advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

            returns = advantages + values
            advantages = whiten(advantages)
            advantages = advantages.detach()

            vpredclipped = clip_by_value(vpred,
                                         values - self.ppo_params["cliprange_value"],
                                         values + self.ppo_params["cliprange_value"])

            vf_losses1 = (vpred - returns) ** 2
            vf_losses2 = (vpredclipped - returns) ** 2
            vf_loss = .5 * torch.mean(torch.max(vf_losses1, vf_losses2))
            vf_clipfrac = torch.mean(torch.gt(vf_losses2, vf_losses1).double())

            ratio = torch.exp(logprob - old_logprobs)

            pg_losses = -advantages * ratio
            pg_losses2 = -advantages * torch.clamp(ratio,
                                                   1.0 - self.ppo_params['cliprange'],
                                                   1.0 + self.ppo_params['cliprange'])

            #             print("advantages")
            #             print(advantages)
            #             print("ratio")
            #             print(ratio)
            #             print("pg_losses1: %s" % str(torch.mean(pg_losses).item()))
            #             print(pg_losses)
            #             print("pg_losses2: %s" % str(torch.mean(pg_losses2).item()))
            #             print(pg_losses2)
            #             print("pg_loss: %s" % str(torch.mean(torch.max(pg_losses, pg_losses2))))
            #             print(torch.max(pg_losses, pg_losses2))

            pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))
            pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses).double())

            loss = pg_loss + self.ppo_params['vf_coef'] * vf_loss

            approxkl = .5 * torch.mean((logprob - old_logprobs) ** 2)
            policykl = torch.mean(logprob - old_logprobs)
            return_mean, return_var = torch.mean(returns), torch.var(returns)
            value_mean, value_var = torch.mean(values), torch.var(values)

            b_pg_loss += pg_loss
            b_vf_loss += vf_loss
            b_loss += loss
            b_approxkl += approxkl
            b_policykl += policykl
            b_pg_clipfrac += pg_clipfrac
            b_advantages_mean += torch.mean(advantages)
            b_return_mean += return_mean
            b_return_var += return_var
            b_mean_vpred += torch.mean(vpred)
            b_error += torch.mean((vpred - returns) ** 2)
            b_vf_clipfrac += vf_clipfrac
            b_value_mean += value_mean
            b_value_var += value_var

        stats = dict(
            loss=dict(policy=b_pg_loss / mini_bs, value=b_vf_loss / mini_bs, total=b_loss / mini_bs),
            policy=dict(approxkl=b_approxkl / mini_bs, policykl=b_policykl / mini_bs, clipfrac=b_pg_clipfrac / mini_bs,
                        advantages_mean=b_advantages_mean / mini_bs),
            returns=dict(mean=b_return_mean / mini_bs, var=b_return_var / mini_bs),
            val=dict(vpred=b_mean_vpred / mini_bs, error=b_error / mini_bs,
                     clipfrac=b_vf_clipfrac / mini_bs, mean=b_value_mean / mini_bs, var=b_value_var / mini_bs),
        )
        return b_pg_loss / mini_bs, self.ppo_params['vf_coef'] * b_vf_loss / mini_bs, flatten_dict(stats)

    def record_step_stats(self, kl_coef, **data):
        """Record training step statistics."""
        all_mean_kl = 0
        bs = self.ppo_params['batch_size']
        for i in range(bs):
            kl = data["logprobs"][i] - data["ref_logprobs"][i]
            mean_kl = torch.mean(torch.sum(kl, axis=-1))
            all_mean_kl += mean_kl

        # kl = data['logprobs'] - data['ref_logprobs']
        # mean_kl = torch.mean(torch.sum(kl, axis=-1))

        stats = {
            'objective/kl': all_mean_kl / bs,  # need this for adaptive kl controller
            'objective/kl_coef': kl_coef,
        }

        for k, v in data['train_stats'].items():
            stats[f'ppo/{k}'] = torch.mean(v, axis=0)
        stats['ppo/val/var_explained'] = 1 - stats['ppo/val/error'] / stats['ppo/returns/var']
        return stats
