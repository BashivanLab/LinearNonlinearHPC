import numpy
import torch
import torch.nn.functional as F
# import time

from torch_ac.algos.base import BaseAlgo

class PPOAlgo(BaseAlgo):
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None, memory_weight_decay=0.0, other_weight_decay=0.0):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        assert self.batch_size % self.recurrence == 0

        memory_params = []
        other_params = []
        for name, param in self.acmodel.named_parameters():
            if 'memory_rnn' in name:
                memory_params.append(param)
            else:
                # other parameters (excluding memory_rnn parameters), e.g. cnn and policy network
                other_params.append(param)

        self.optimizer = torch.optim.Adam([
            {'params': memory_params, 'weight_decay': memory_weight_decay},
            {'params': other_params, 'weight_decay': other_weight_decay}
        ], lr, eps=adam_eps)

        # self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
        self.batch_num = 0

    def update_parameters(self, exps):
        # Collect experiences
        # start_time = time.time()
        # time_0 = time.time()

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            # print(f'start epoch {_} after : {time.time() - time_0} (seconds)')
            # time_0 = time.time()
            # __ = 0

            for inds in self._get_batches_starting_indexes():

                # time_1 = time.time()
                # __ += 1

                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # Initialize memory
                if 'epn' in self.acmodel.name:
                    memory = exps.memory[inds]
                    memory_index = exps.memory_index[inds]
                elif self.acmodel.recurrent:
                    rnn_memory = exps.rnn_memory[inds]

                # time_2 = time.time()
                # print(f'    Time in epoch {_} batch {__}. Init memory: {time_2 - time_1} (seconds)')

                for i in range(self.recurrence):
                    # Create a sub-batch of experience

                    sb = exps[inds + i]

                    # Compute loss

                    if 'epn' in self.acmodel.name:
                        dist, value, memory, memory_index = self.acmodel(sb.obs, memory * sb.mask.unsqueeze(2), (memory_index * sb.mask).squeeze())
                    elif self.acmodel.recurrent:
                        dist, value, rnn_memory = self.acmodel(sb.obs, rnn_memory * sb.mask)
                    else:
                        dist, value = self.acmodel(sb.obs)

                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch

                    if self.acmodel.recurrent and i < self.recurrence - 1:
                        exps.rnn_memory[inds + i + 1] = rnn_memory.detach()
                    # TODO does make it sense to do that? it should only change one row in the memory and that row becomes: slightly different image embedding, same: reward, action, task-index. Help for staleness?
                    if 'epn' in self.acmodel.name: #and exps.memory.shape[0] < inds+i+1 and (inds+i+1) % self.num_frames_per_proc != 0: # (inds+i+1) % self.num_frames_per_proc != 0 -> check if the next index is the following frame from that experience or it belongs to another parallelenv
                        # Check if the next index is the following frame from that experience or it belongs to another parallelenv
                        # shape of experiemnts: [parallelenv_0-step_0, parallelenv_0-step_1, ..., parallelenv_0-step_{num_frames_per_proc}, parallelenv_1-step_0, ...]
                        # If inds+i+1 is from the same parallelenv, but it's a different episode, it's ok to update the memory anyway because it will be zero'd with the mask.
                        inds_of_frames_with_valid_next_frame = inds[((inds+i+1) % self.num_frames_per_proc != 0)]
                        exps.memory[inds_of_frames_with_valid_next_frame + i + 1] = memory[(inds+i+1) % self.num_frames_per_proc != 0].detach()


                # time_3 = time.time()
                # print(f'    Time in epoch {_} batch {__}. Forward pass to create sub-batch: {time_3 - time_2} (seconds)')

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic

                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # print(f'    Time in epoch {_} batch {__}. Backpropagation and optimizer step: {time.time() - time_3} (seconds) \n')

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)

        # Log some values

        logs = {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "grad_norm": numpy.mean(log_grad_norms)
        }
        # print('update time:', time.time() - start_time)
        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_frames_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
