# Modified from https://github.com/lvwerra/trl/blob/master/trl


# Cell
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
import torch
import collections
import time
import random

from utils import concat_past

from ppo_utils import (logprobs_from_logits,
                         whiten,
                         clip_by_value,
                         entropy_from_logits,
                         flatten_dict,
                         average_torch_dicts,
                         stats_to_np,
                         stack_dicts,
                         add_suffix)

# Cell

class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """
    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult

# Cell

class FixedKLController:
    """Fixed KL controller."""
    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass

# Cell

class PPOTrainer:
    """
    The PPO_trainer uses Proximal Policy Optimization to optimise language models.
    """

    default_params = {
        "lr": 1.41e-5,
        "adap_kl_ctrl": True,
        "init_kl_coef":0.2,
        "target": 6,
        "horizon":10000,
        "gamma":1,
        "lam":0.95,
        "cliprange": .2,
        "cliprange_value":.2,
        "vf_coef":.1,
        "batch_size": 256,
        "forward_batch_size": 16,
        "ppo_epochs": 4,
        "ppo_mini_batch-size": 4,
    }

    def __init__(self, model, optimizer, **ppo_params):
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
        # self.optimizer = Adam(model.parameters(), lr=self.ppo_params['lr'])
        self.optimizer = optimizer

        self.kl_ctl = AdaptiveKLController(self.ppo_params['init_kl_coef'],
                                           self.ppo_params['target'],
                                           self.ppo_params['horizon'])


    def step(self, all_c_p_tensors, all_c_p_lengths, all_c_lengths, all_scores):
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

        # gen_len = response.shape[1]
        # model_input = torch.cat((query, response), axis=1)

        t = time.time()
        # logprobs, ref_logprobs, values = self.batched_forward_pass(model_input, gen_len)
        # timing['time/ppo/forward_pass'] = time.time()-t
        #
        # t = time.time()
        # rewards, non_score_reward, kl_coef = self.compute_rewards(scores, logprobs, ref_logprobs)
        # timing['time/ppo/compute_rewards'] = time.time()-t
        logprobs, ref_logprobs, values, rewards, non_score_reward, kl_coef, real_p_tensors, real_c_p_tensors, real_c_p_lengths, real_c_lengths = self.batched_trigger_forward_pass(
            all_c_p_tensors, all_c_p_lengths, all_c_lengths, all_scores)
        timing['time/ppo/batched_trigger_forward'] = time.time()-t


        t = time.time()

        all_stats = []
        idxs = list(range(bs))
        # for _ in range(self.ppo_params['ppo_epochs']):
        #     random.shuffle(idxs)
        #     for i in range(bs):
        #         idx = idxs[i]
        #         train_stats = self.train_minibatch(logprobs[idx:idx+1], values[idx:idx+1],
        #                                            rewards[idx:idx+1], query[idx:idx+1],
        #                                            response[idx:idx+1], model_input[idx:idx+1])
        #         all_stats.append(train_stats)
        for _ in range(self.ppo_params['ppo_epochs']):
            random.shuffle(idxs)
            for i in range(bs // mini_bs):
                b_idx = idxs[i*mini_bs:(i+1)*mini_bs]
                b_logprobs, b_values, b_rewards, b_p_tensors, b_c_p_tensors, b_c_p_lengths, b_c_lengths = \
                    list(), list(), list(), list(), list(), list(), list()
                for b_idx_i in b_idx:
                    b_logprobs.append(logprobs[b_idx_i])
                    b_values.append(values[b_idx_i])
                    b_rewards.append(rewards[b_idx_i])
                    b_p_tensors.append(real_p_tensors[b_idx_i])
                    b_c_p_tensors.append(real_c_p_tensors[b_idx_i])
                    b_c_p_lengths.append(real_c_p_lengths[b_idx_i])
                    b_c_lengths.append(real_c_lengths[b_idx_i])

                # train_stats = self.train_minibatch(logprobs[idx:idx+1], values[idx:idx+1],
                #                                    rewards[idx:idx+1], query[idx:idx+1],
                #                                    response[idx:idx+1], model_input[idx:idx+1])

                train_stats = self.train_minibatch(b_logprobs, b_values, b_rewards, b_p_tensors, b_c_p_tensors, b_c_p_lengths, b_c_lengths)

                all_stats.append(train_stats)

        timing['time/ppo/optimize_step'] = time.time()-t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats['policy/advantages'] = torch.flatten(train_stats['policy/advantages']).unsqueeze(0)
        train_stats['policy/ratio'] = torch.flatten(train_stats['policy/ratio']).unsqueeze(0)

        stats = self.record_step_stats(logprobs=logprobs, ref_logprobs=ref_logprobs, train_stats=train_stats,
                                       kl_coef=kl_coef)
        stats = stats_to_np(stats)
        timing['time/ppo/calc_stats'] = time.time()-t

        self.kl_ctl.update(stats['objective/kl'], self.ppo_params['batch_size'])

        timing['time/ppo/total'] = time.time()-t0
        stats.update(timing)
        return stats

    def batched_forward_pass(self, model_input, gen_len):
        """Calculate model outputs in multiple batches."""
        bs = self.ppo_params['batch_size']
        fbs = self.ppo_params['forward_batch_size']
        logprobs = []
        ref_logprobs = []
        values = []

        for i in range(int(self.ppo_params['batch_size']/fbs)):
            m_input = model_input[i*fbs:(i+1)*fbs]
            logits, _, v = self.model(m_input)
            ref_logits, _, _ = self.ref_model(m_input)

            values.append(v[:, -gen_len-1:-1].detach())
            logprobs.append(logprobs_from_logits(logits[:,:-1,:], m_input[:,1:])[:, -gen_len:].detach())
            ref_logprobs.append(logprobs_from_logits(ref_logits[:,:-1,:], m_input[:,1:])[:, -gen_len:].detach())

        return torch.cat(logprobs), torch.cat(ref_logprobs), torch.cat(values)

    def get_trigger_forward_pass(self, model_input, is_ref=False):
        reset_pos_emb = self.ppo_params["reset_pos_emb"]
        num_of_triggers = self.ppo_params["num_of_triggers"]
        trigger_format = self.ppo_params["trigger_format"]
        TRIGGER_POSITION_ID = self.ppo_params["TRIGGER_POSITION_ID"]
        device = self.device
        past = self.lm_bos_output["past_key_values"]
        batch_size, batch_max_length = model_input.shape

        if reset_pos_emb:
            t_position_ids = torch.ones(batch_size, num_of_triggers).to(torch.long).to(device) * TRIGGER_POSITION_ID
        else:
            t_position_ids = None

        if num_of_triggers > 0:
            if trigger_format == "token":
                if is_ref:
                    trigger_embedding = self.model.ref_ori_trigger_embedding.repeat(batch_size, 1, 1)
                else:
                    trigger_embedding = self.model.ori_trigger_embedding.repeat(batch_size, 1, 1)
                lm_trigger_output = self.model(inputs_embeds=trigger_embedding, position_ids=t_position_ids)
                trigger_key_values = lm_trigger_output["past_key_values"]
            else:
                if is_ref:
                    trigger_key_values = expand_past(model.ori_trigger_key_values, num_layers, batch_size)
                else:
                    trigger_key_values = expand_past(model.ref_ori_trigger_key_values, num_layers, batch_size)
            past = concat_past(past, trigger_key_values, num_layers)

        if reset_pos_emb:
            p_position_ids = torch.arange(1, batch_max_length, dtype=torch.long, device=device)
            p_position_ids = p_position_ids.unsqueeze(0).repeat(batch_size, 1)
        else:
            p_position_ids = None

        logits, _, v = self.model(model_input[:, 1:], past_key_values=past, position_ids=p_position_ids)  # model_input[:, 0] is BOS

        return logits, _, v

    def batched_trigger_forward_pass(self, all_c_p_tensors, all_c_p_lengths, all_c_lengths, all_scores):
        # combines batched_forward_pass and compute_rewards
        logprobs, ref_logprobs, values = list(), list(), list()
        rewards, non_score_rewards = list(), list()
        real_p_tensors, real_c_p_tensors, real_c_p_lengths, real_c_lengths = list(), list(), list()
        for i in range(len(all_c_lengths)):  # number of minibatches
            model_input = all_c_p_tensors[i]
            logits, _, v = self.get_trigger_forward_pass(model_input)
            ref_logits, _, _ = self.get_trigger_forward_pass(model_input, is_ref=True)
            lp = logprobs_from_logits(logits[:, :-1, :], model_input[:, 1:])
            ref_lp = logprobs_from_logits(ref_logits[:, :-1, :], model_input[:, 1:])

            for j in range(len(all_c_lengths[i])):  # loop through the minibatch to get the real indices
                start = all_c_lengths[i][j] - 1
                end = all_c_p_lengths[i][j] - 1
                values.append(v[j:j+1, start:end].detach())
                ij_logprob = lp[j:j+1, start:end].detach()
                ij_ref_logprob = ref_lp[j:j+1, start:end].detach()
                logprobs.append(ij_logprob)
                ref_logprobs.append(ij_ref_logprob)
                real_p_tensors.append(all_c_p_tensors[i][j:j+1, start+1:end+1])
                real_c_p_tensors.append(all_c_p_tensors[i][j:j+1, :end+1])
                real_c_p_lengths.append(end + 1)
                real_c_lengths.append(start + 1)

                # compute rewards
                ij_reward, ij_non_score_reward, kl_coef = self.compute_rewards(all_scores[i][j], ij_logprob, ij_ref_logprob)
                rewards.append(ij_reward)
                non_score_rewards.append(ij_non_score_reward)

        return logprobs, ref_logprobs, values, rewards, non_score_rewards, kl_coef, real_p_tensors, real_c_p_tensors, real_c_p_lengths, real_c_lengths

    # def train_minibatch(self, logprobs, values, rewards, query, response, model_input):
    def train_minibatch(self, b_logprobs, b_values, b_rewards, b_p_tensors, b_c_p_tensors, b_c_p_lengths, b_c_lengths):
        """Train one PPO minibatch"""
        loss_p, loss_v, train_stats  = self.loss(b_logprobs, b_values, b_rewards, b_p_tensors, b_c_p_tensors, b_c_p_lengths, b_c_lengths)
        loss = loss_p + loss_v
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
    def loss(self, old_b_logprobs, b_values, b_rewards, b_p_tensors, b_c_p_tensors, b_c_p_lengths, b_c_lengths):
        """Calculate policy and value losses."""
        # lastgaelam = 0
        # advantages_reversed = []

        # Note: values, old_logprobs are for prompts only (without context)

        # logits, _, vpred = self.model(model_input)
        # logprob = logprobs_from_logits(logits[:,:-1,:], model_input[:, 1:])
        #
        # #only the generation part of the values/logprobs is needed
        # logprob, vpred = logprob[:, -gen_len:], vpred[:,-gen_len-1:-1]

        mini_bs = self.ppo_params["ppo_mini_batch_size"]
        # pad mini batch for forward path
        batch_max_length = max(len(c_p_t) for c_p_t in b_c_p_tensors)
        padded_tensors = list()
        for c_p_t in b_c_p_tensors:
            padded_tensors.append(torch.cat(c_p_t, torch.ones(1, batch_max_length - c_p_t.shape[1], dtype=torch.long, device=device) * self.ppo_params["padding_token"]), dim=1)
        padded_tensors = torch.cat(padded_tensors)  # mini_bs x batch_max_length
        b_logits, _, b_vpred = self.model(padded_tensors)
        b_logprob = logprobs_from_logits(b_logits[:, :-1, :], padded_tensors[:, 1:])

        b_pg_loss, b_vf_loss, b_loss, b_entropy, b_approxkl, b_policykl, b_pg_clipfrac,\
        b_advantages_mean, b_return_mean, b_return_var, b_vpred, b_error, b_vf_clipfrac, b_value_mean, b_value_var = \
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

        for j in range(mini_bs):

            start = b_c_lengths[j] - 1
            end = b_c_p_lengths[j] - 1

            logprob = b_logprob[:, start:end]
            vpred = b_vpred[:, start:end]
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

            vf_losses1 = (vpred - returns)**2
            vf_losses2 = (vpredclipped - returns)**2
            vf_loss = .5 * torch.mean(torch.max(vf_losses1, vf_losses2))
            vf_clipfrac =  torch.mean(torch.gt(vf_losses2, vf_losses1).double())

            ratio = torch.exp(logprob - old_logprobs)

            pg_losses = -advantages * ratio
            pg_losses2 = -advantages * torch.clamp(ratio,
                                                   1.0 - self.ppo_params['cliprange'],
                                                   1.0 + self.ppo_params['cliprange'])

            pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))
            pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses).double())

            loss = pg_loss + self.ppo_params['vf_coef'] * vf_loss

            entropy = torch.mean(entropy_from_logits(logits))
            approxkl = .5 * torch.mean((logprob - old_logprobs)**2)
            policykl = torch.mean(logprob - old_logprobs)
            return_mean, return_var = torch.mean(returns), torch.var(returns)
            value_mean, value_var = torch.mean(values), torch.var(values)

            b_pg_loss += pg_loss
            b_vf_loss += vf_loss
            b_loss += loss
            b_entropy += entropy
            b_approxkl += approxkl
            b_policykl += policykl
            b_pg_clipfrac += pg_clipfrac
            b_advantages_mean += torch.mean(advantages)
            b_return_mean += return_mean
            b_return_var += return_var
            b_vpred += torch.mean(vpred)
            b_error += torch.mean((vpred - returns) ** 2)
            b_vf_clipfrac += vf_clipfrac
            b_value_mean += value_mean
            b_value_var += value_var

        # stats = dict(
        #     loss=dict(policy=pg_loss, value=vf_loss, total=loss),
        #     policy=dict(entropy=entropy, approxkl=approxkl,policykl=policykl, clipfrac=pg_clipfrac,
        #                 advantages=advantages, advantages_mean=torch.mean(advantages), ratio=ratio),
        #     returns=dict(mean=return_mean, var=return_var),
        #     val=dict(vpred=torch.mean(vpred), error=torch.mean((vpred - returns) ** 2),
        #              clipfrac=vf_clipfrac, mean=value_mean, var=value_var),
        # )
        # return pg_loss, self.ppo_params['vf_coef'] * vf_loss, flatten_dict(stats)
        stats = dict(
            loss=dict(policy=b_pg_loss/mini_bs, value=b_vf_loss/mini_bs, total=b_loss/mini_bs),
            policy=dict(entropy=b_entropy/mini_bs, approxkl=b_approxkl/mini_bs, policykl=b_policykl/mini_bs, clipfrac=b_pg_clipfrac/mini_bs,
                        advantages_mean=b_advantages_mean/mini_bs),
            returns=dict(mean=b_return_mean/mini_bs, var=b_return_var/mini_bs),
            val=dict(vpred=b_vpred/mini_bs, error=b_error/mini_bs,
                     clipfrac=b_vf_clipfrac/mini_bs, mean=b_value_mean/mini_bs, var=b_value_var/mini_bs),
        )
        return b_pg_loss/mini_bs, self.ppo_params['vf_coef'] * b_vf_loss/mini_bs, flatten_dict(stats)


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