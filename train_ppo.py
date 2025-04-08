import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Tuple, List, Any
from free_range_zoo.utils.agent import Agent
from free_range_zoo.envs import wildfire_v0
from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0

ROLLOUT_STEPS = 256
PPO_EPOCHS = 4
MINI_BATCH_SIZE = 256
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 3e-4
VF_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5
TASK_DIM = 8
MODE_DIM = 2
MAX_TASKS = 20


def flatten_obs(obs_tuple, env_idx) -> torch.Tensor:
    tensordict_part, _ = obs_tuple
    self_part = tensordict_part["self"][env_idx].flatten()
    others_part = tensordict_part["others"][env_idx].flatten()
    tasks_part = tensordict_part["tasks"][env_idx].flatten()
    desired_len = MAX_TASKS * 4
    current_len = tasks_part.numel()
    if current_len < desired_len:
        pad = torch.zeros(desired_len - current_len, device=tasks_part.device, dtype=tasks_part.dtype)
        tasks_part = torch.cat([tasks_part, pad], dim=0)
    else:
        tasks_part = tasks_part[:desired_len]
    return torch.cat([self_part, others_part, tasks_part], dim=0)


class ActorCritic(Agent, nn.Module):

    def __init__(self, agent_name: str, parallel_envs: int, obs_dim: int, task_dim: int, mode_dim: int, device: torch.device):
        Agent.__init__(self, agent_name, parallel_envs)
        nn.Module.__init__(self)

        self.agent_name = agent_name
        self.obs_dim = obs_dim
        self.device = device
        self.current_obs = None

        hidden = 256
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.task_head = nn.Linear(hidden, task_dim)
        self.mode_head = nn.Linear(hidden, mode_dim)
        self.value_head = nn.Linear(hidden, 1)
        self.to(device)

    def forward(self, obs: torch.Tensor):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        task_logits = self.task_head(x)
        mode_logits = self.mode_head(x)
        value = self.value_head(x).squeeze(-1)
        return task_logits, mode_logits, value

    def observe(self, observation: Dict[str, Any]) -> None:
        tensordict_part, _ = observation
        obs_list = []
        for b in range(self.parallel_envs):
            flat_obs = flatten_obs((tensordict_part, {}), b)
            obs_list.append(flat_obs)
        self.current_obs = torch.stack(obs_list).to(self.device)

    def act(self, action_space: any) -> List[List[int]]:
        if self.current_obs is None:
            return [[-1, -1] for _ in range(self.parallel_envs)]
        with torch.no_grad():
            task_logits, mode_logits, _ = self.forward(self.current_obs)
            raw_task = task_logits.argmax(dim=1)
            raw_mode = mode_logits.argmax(dim=1)

        agent_idx = None
        if hasattr(action_space, 'agent_task_count'):
            agent_idx = action_space.agent_name_mapping[self.agent_name]
            valid_tasks = action_space.agent_task_count[agent_idx].cpu()
        else:
            valid_tasks = torch.ones(self.parallel_envs, dtype=torch.int32, device=self.device)
        actions = clamp_action(raw_task, raw_mode, valid_tasks)
        return actions.tolist()

    def get_action_and_value(self, obs: torch.Tensor):
        task_logits, mode_logits, value = self.forward(obs)
        dist_task = torch.distributions.Categorical(logits=task_logits)
        dist_mode = torch.distributions.Categorical(logits=mode_logits)
        task_action = dist_task.sample()
        mode_action = dist_mode.sample()
        logprob_task = dist_task.log_prob(task_action)
        logprob_mode = dist_mode.log_prob(mode_action)
        total_logprob = logprob_task + logprob_mode
        return task_action, mode_action, total_logprob, value, (dist_task, dist_mode)

    def evaluate_actions(self, obs: torch.Tensor, task_action: torch.Tensor, mode_action: torch.Tensor):
        task_logits, mode_logits, value = self.forward(obs)
        dist_task = torch.distributions.Categorical(logits=task_logits)
        dist_mode = torch.distributions.Categorical(logits=mode_logits)
        logprob_task = dist_task.log_prob(task_action)
        logprob_mode = dist_mode.log_prob(mode_action)
        total_logprob = logprob_task + logprob_mode
        entropy = dist_task.entropy() + dist_mode.entropy()
        return total_logprob, value, entropy.mean()


class RolloutBuffer:

    def __init__(self, num_envs: int, obs_dim: int, rollout_steps: int):
        self.obs = torch.zeros((rollout_steps + 1, num_envs, obs_dim), dtype=torch.float)
        self.task_actions = torch.zeros((rollout_steps, num_envs), dtype=torch.long)
        self.mode_actions = torch.zeros((rollout_steps, num_envs), dtype=torch.long)
        self.log_probs = torch.zeros((rollout_steps, num_envs), dtype=torch.float)
        self.rewards = torch.zeros((rollout_steps, num_envs), dtype=torch.float)
        self.values = torch.zeros((rollout_steps + 1, num_envs), dtype=torch.float)
        self.dones = torch.zeros((rollout_steps, num_envs), dtype=torch.bool)
        self.step = 0
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.rollout_steps = rollout_steps

    def insert(self, obs: torch.Tensor, task_a: torch.Tensor, mode_a: torch.Tensor,
               logp: torch.Tensor, rew: torch.Tensor, val: torch.Tensor, done: torch.Tensor):
        idx = self.step
        self.obs[idx] = obs
        self.task_actions[idx] = task_a
        self.mode_actions[idx] = mode_a
        self.log_probs[idx] = logp
        self.rewards[idx] = rew
        self.values[idx] = val
        self.dones[idx] = done
        self.step += 1

    def finish(self, last_value: torch.Tensor, gamma: float, lam: float):
        # last_value = last_value.view(-1, 1)
        # dont need to send last_value anymore bc its in already
        advantages = torch.zeros((self.rollout_steps, self.num_envs), dtype=torch.float)
        gae = torch.zeros((self.num_envs,), dtype=torch.float)
        for t in reversed(range(self.step)):
            delta = (self.rewards[t]
                     + gamma * self.values[t + 1] * (~self.dones[t]).float()
                     - self.values[t])
            gae = delta + gamma * lam * (~self.dones[t]).float() * gae
            advantages[t] = gae
        returns = advantages + self.values[:-1]
        return advantages, returns


def clamp_action(raw_task: torch.Tensor, raw_mode: torch.Tensor, valid_tasks: torch.Tensor) -> torch.Tensor:
    # wtf please vectorize this
    envs = raw_task.size(0)
    final = torch.zeros((envs, 2), dtype=torch.int64, device=raw_task.device)
    for i in range(envs):
        k = valid_tasks[i].item()
        if k <= 1:
            final[i, 0] = 0
            final[i, 1] = -1 if k == 0 else (-1 if raw_mode[i].item() == 1 else 0)
        else:
            a_raw = raw_task[i].item()
            mapped = int((a_raw / (TASK_DIM - 1)) * (k - 1))
            final[i, 0] = mapped
            m = raw_mode[i].item()
            final[i, 1] = -1 if m == 1 else 0
    return final


def ppo_update(policy: ActorCritic, optimizer: optim.Optimizer, buffer: RolloutBuffer):
    advantages, returns = buffer.finish(buffer.values[buffer.step], gamma=GAMMA, lam=LAMBDA)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    num_steps = buffer.step
    batch_size = num_steps * buffer.num_envs
    obs_flat = buffer.obs[:num_steps].reshape(batch_size, -1)
    pol_device = next(policy.parameters()).device
    obs_flat = obs_flat.to(pol_device)
    task_flat = buffer.task_actions[:num_steps].reshape(batch_size)
    mode_flat = buffer.mode_actions[:num_steps].reshape(batch_size)
    old_logps_flat = buffer.log_probs[:num_steps].reshape(batch_size)
    returns_flat = returns[:num_steps].reshape(batch_size)
    advs_flat = advantages[:num_steps].reshape(batch_size)
    for _ in range(PPO_EPOCHS):
        idxs = torch.randperm(batch_size)
        start = 0
        while start < batch_size:
            end = min(start + MINI_BATCH_SIZE, batch_size)
            bidx = idxs[start:end]
            batch_obs = obs_flat[bidx]
            batch_task = task_flat[bidx].to(pol_device)
            batch_mode = mode_flat[bidx].to(pol_device)
            batch_old_lp = old_logps_flat[bidx].to(pol_device)
            batch_ret = returns_flat[bidx].to(pol_device)
            batch_adv = advs_flat[bidx].to(pol_device)
            new_logps, vpred, entropy = policy.evaluate_actions(batch_obs, batch_task, batch_mode)
            ratio = torch.exp(new_logps - batch_old_lp)
            surr1 = ratio * batch_adv
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * batch_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(vpred, batch_ret)
            loss = policy_loss + VF_COEF * value_loss - ENTROPY_COEF * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            start = end


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="marl_ppo_output")
    parser.add_argument("--parallel_envs", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config, "rb") as f:
        configuration = pickle.load(f)

    # Create environment
    env = wildfire_v0.parallel_env(
        parallel_envs=args.parallel_envs,
        max_steps=100,
        configuration=configuration,
        device=device,
        buffer_size=50,
        show_bad_actions=False,
        observe_other_power=False,
        observe_other_suppressant=False,
        log_directory=os.path.join(args.output_dir, "env_logs"),
        override_initialization_check=True
    )
    env = action_mapping_wrapper_v0(env)
    agents = env.agents
    print("Agents:", agents)

    obs_dict, _ = env.reset(seed=0)
    example_agent = agents[0]
    tensordict_part, _ = obs_dict[example_agent]
    example_flat = flatten_obs((tensordict_part, {}), 0)
    obs_dim = example_flat.size(0)
    # update to use api

    policies = {}
    optimizers = {}
    for ag in agents:
        pol = ActorCritic(
            agent_name=ag,
            parallel_envs=args.parallel_envs,
            obs_dim=obs_dim,
            task_dim=TASK_DIM,
            mode_dim=MODE_DIM,
            device=device
        )
        policies[ag] = pol
        optimizers[ag] = optim.Adam(pol.parameters(), lr=LR)

    for ep in range(args.episodes):
        obs_dict, info = env.reset(seed=ep)
        buffers = {}
        for ag in agents:

            buffers[ag] = RolloutBuffer(num_envs=args.parallel_envs, obs_dim=obs_dim, rollout_steps=ROLLOUT_STEPS)
        current_obs = {}
        for ag in agents:
            tensordict_part, _ = obs_dict[ag]
            stack_list = []
            for b in range(args.parallel_envs):
                flat_obs = flatten_obs((tensordict_part, {}), b)
                stack_list.append(flat_obs)
            current_obs[ag] = torch.stack(stack_list).to(device)
        for ag in agents:
            with torch.no_grad():
                _, _, val, _, _ = policies[ag].get_action_and_value(current_obs[ag])
            buffers[ag].obs[0] = current_obs[ag].cpu()
            buffers[ag].values[0] = val.cpu()

        step_count = 0
        while step_count < ROLLOUT_STEPS:
            if torch.all(env.finished):
                break

            raw_tasks_dict = {}
            raw_modes_dict = {}
            logps_dict = {}
            values_dict = {}
            for ag in agents:
                pol = policies[ag]
                obs_t = current_obs[ag]
                with torch.no_grad():
                    raw_task, raw_mode, logp, val, _ = pol.get_action_and_value(obs_t)
                raw_tasks_dict[ag] = raw_task.cpu()
                raw_modes_dict[ag] = raw_mode.cpu()
                logps_dict[ag] = logp.cpu()
                values_dict[ag] = val.cpu()

            final_actions_for_env = {}
            for ag in agents:
                #illegal?
                agent_idx = env.unwrapped.agent_name_mapping[ag]
                agent_task_counts = env.unwrapped.agent_task_count[agent_idx].cpu()
                env_actions = []
                for b in range(args.parallel_envs):
                    k = agent_task_counts[b].item()
                    # this is probably horrible
                    valid_tasks = torch.tensor([k], device=raw_tasks_dict[ag].device)
                    a2d = clamp_action(
                        raw_tasks_dict[ag][b].unsqueeze(0),
                        raw_modes_dict[ag][b].unsqueeze(0),
                        valid_tasks
                    )
                    env_actions.append(a2d[0])
                env_actions = torch.stack(env_actions, dim=0)
                final_actions_for_env[ag] = env_actions.to(device)

            next_obs_dict, rew_dict, term_dict, trunc_dict, infos = env.step(final_actions_for_env)
            r_team = torch.zeros(args.parallel_envs, dtype=torch.float32, device=device)
            for ag in agents:
                r_team += rew_dict[ag]
            for ag in agents:
                # probably wrong
                buffers[ag].insert(
                    obs=buffers[ag].obs[step_count],
                    task_a=raw_tasks_dict[ag],
                    mode_a=raw_modes_dict[ag],
                    logp=logps_dict[ag],
                    rew=r_team,
                    val=values_dict[ag],
                    done=(term_dict[ag] | trunc_dict[ag])
                )
            for ag in agents:
                tensordict_part, _ = next_obs_dict[ag]
                stack_list = []
                for b in range(args.parallel_envs):
                    fl = flatten_obs((tensordict_part, {}), b)
                    stack_list.append(fl)
                current_obs[ag] = torch.stack(stack_list).to(device)
            obs_dict = next_obs_dict
            for ag in agents:
                with torch.no_grad():
                    _, _, vnext, _, _ = policies[ag].get_action_and_value(current_obs[ag])
                buffers[ag].values[step_count + 1] = vnext.cpu()
                buffers[ag].obs[step_count + 1] = current_obs[ag].cpu()
            step_count += 1

        for ag in agents:
            ppo_update(policies[ag], optimizers[ag], buffers[ag])
        ep_team_return = buffers[agents[0]].rewards.sum(dim=0).mean().item()
        print(f"[Episode {ep}] Mean team return: {ep_team_return:.3f}")

    for ag in agents:
        path = os.path.join(args.output_dir, f"{ag}_final.pt")
        torch.save(policies[ag].state_dict(), path)
    print("Done training.")


if __name__ == "__main__":
    main()
