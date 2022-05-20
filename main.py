import numpy as np
import torch
import os
import gym
from dotmap import DotMap
from pathlib import Path
from tensorboardX import SummaryWriter
import wandb as wb

import envs
import envs.gymmb
from algo.td3 import TD3
from components.dynamics import Model
# from components.reward_dynamics import RewardModel
from components.arguments import common_args, policy_function, value_function, dynamics_model
from components.env_loop import EnvLoop
from components.buffer import Buffer
from components.normalizer import TransitionNormalizer
from utils.wrappers import BoundedActionsEnv, IsDoneEnv, MuJoCoCloseFixWrapper, RecordedEnv
from utils.misc import to_np, EpisodeStats
from utils.radam import RAdam


def get_random_agent(d_action, device):
    class RandomAgent:
        @staticmethod
        def get_action(states, deterministic=False):
            return torch.rand(size=(states.shape[0], d_action), device=device) * - 1
    return RandomAgent()


def get_deterministic_agent(agent):
    class DeterministicAgent:
        @staticmethod
        def get_action(states):
            return agent.get_action(states, deterministic=True)
    return DeterministicAgent()


def get_env(env_name, record=False):
    env = gym.make(env_name)
    env = BoundedActionsEnv(env)

    env = IsDoneEnv(env)
    env = MuJoCoCloseFixWrapper(env)
    if record:
        env = RecordedEnv(env)

    env.seed(np.random.randint(np.iinfo(np.uint32).max))
    if hasattr(env.action_space, 'seed'):  # Only for more recent gym
        env.action_space.seed(np.random.randint(np.iinfo(np.uint32).max))
    if hasattr(env.observation_space, 'seed'):  # Only for more recent gym
        env.observation_space.seed(np.random.randint(np.iinfo(np.uint32).max))

    return env


class MainLoopTraining:
    def __init__(self, logger, args):
        self.step_i = 0
        # env_config
        tmp_env = gym.make(args.env_name)
        # Cheetah, Pusher, Swimmer, Pendulum --> is_done always return false
        self.is_done = tmp_env.unwrapped.is_done
        # eval task: default standard
        self.eval_tasks = {args.task_name: tmp_env.tasks()[args.task_name]}
        # Exploitation_task: reward computation
        self.exploitation_task = tmp_env.tasks()[args.task_name]
        self.d_state = tmp_env.observation_space.shape[0]
        self.d_action = tmp_env.action_space.shape[0]
        self.max_episode_steps = tmp_env.spec.max_episode_steps
        del tmp_env
        if args.use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.logger = logger
        # Wrapped Env
        self.env_loop = EnvLoop(get_env, env_name=args.env_name, render=args.render)

        # Transition Model
        self.model = Model(
            d_state=self.d_state, d_action=self.d_action, n_units=args.model_n_units, n_layers=args.model_n_layers,
            ensemble_size=args.model_ensemble_size, activation=args.model_activation,
            device=self.device
        )
        self.model_optimizer = RAdam(self.model.parameters(), lr=args.model_lr, weight_decay=args.model_weight_decay)
        # self.reward_model = RewardModel(
        #     d_state=self.d_state, d_action=self.d_action, n_units=256, n_layers=3,
        #     activation=args.model_activation, device=self.device
        # )
        # self.reward_model_optimizer = RAdam(self.reward_model.parameters(),
        #                                     lr=args.model_lr, weight_decay=args.model_weight_decay)

        # Buffer and Normalizer
        self.buffer = Buffer(self.d_state, self.d_action, args.n_total_steps)
        if args.normalize_data:
            self.buffer.setup_normalizer(TransitionNormalizer(self.d_state, self.d_action, self.device))

        # Agent
        self.agent = TD3(
            d_state=self.d_state, d_action=self.d_action, device=self.device, gamma=args.gamma, tau=args.tau,
            policy_lr=args.policy_lr, value_lr=args.value_lr,
            value_loss=args.value_loss, value_n_layers=args.value_n_layers, value_n_units=args.value_n_units,
            value_activation=args.value_activation,
            policy_n_layers=args.policy_n_layers, policy_n_units=args.policy_n_units,
            policy_activation=args.policy_activation, grad_clip=args.grad_clip, policy_delay=args.policy_delay,
            expl_noise=args.td3_expl_noise
        )

        self.stats = EpisodeStats(self.eval_tasks)
        self.last_avg_eval_score = None
        self.random_agent = get_random_agent(self.d_action, self.device)

        self.args = args
        self.plot_rews, self.plot_steps = [], []

    def evaluate_on_task(self):
        # print(f"MainLoopStep {self.step_i} | evaluate | evaluating model for task ...")
        env = get_env(self.args.env_name, record=False)
        task = env.unwrapped.tasks()[self.args.task_name]
        env.close()

        episode_returns, episode_length = [], []
        env_loop = EnvLoop(get_env, env_name=self.args.env_name, render=False)
        agent = get_deterministic_agent(self.agent)

        # Test agent on real environment by running an episode
        for ep_i in range(self.args.n_eval_episodes):
            with torch.no_grad():
                states, actions, next_states = env_loop.episode(agent)
                rewards = task(states, actions, next_states)

            ep_return = rewards.sum().item()
            ep_len = len(rewards)
            episode_returns.append(ep_return)
            episode_length.append(ep_len)
        env_loop.close()

        avg_ep_return = np.mean(episode_returns)
        avg_ep_length = np.mean(episode_length)
        # print(f"MainLoopStep {self.step_i} | evaluate | AverageReturns {avg_ep_return: 5.2f} | AverageLength {avg_ep_length: 5.2f}")

        return avg_ep_return

    def train(self):
        self.step_i += 1

        behavior_agent = self.random_agent if self.step_i <= self.args.n_warm_up_steps else self.agent
        with torch.no_grad():
            action = behavior_agent.get_action(self.env_loop.state, deterministic=False).to('cpu')

        state, next_state, done = self.env_loop.step(to_np(action))
        reward = self.exploitation_task(state, action, next_state).item()
        self.buffer.add(state, action, next_state, torch.from_numpy(np.array([[reward]], dtype=np.float32)))
        self.stats.add(state, action, next_state, done)

        # Update Episodic Memory with Transition Model
        if done:
            for task_name in self.eval_tasks:
                last_ep_return = self.stats.ep_returns[task_name][-1]
                last_ep_length = self.stats.ep_lengths[task_name][-1]
                # print(f"MainLoopStep {self.step_i} | train | EpReturns {last_ep_return: 5.2f} | EpLength {last_ep_length: 5.2f}")
                wb.log({"TrainingEpReturn": last_ep_return}, step=self.step_i)

        # Training Dynamics Model
        # -------------------------------
        if (self.args.model_training_freq is not None and self.args.model_training_n_batches > 0
                and self.step_i % self.args.model_training_freq == 0):
            self.model.setup_normalizer(self.buffer.normalizer)

            loss = np.nan
            batch_i = 0
            while batch_i < self.args.model_training_n_batches:
                losses = self._train_model_epoch()
                batch_i += len(losses)
                loss = np.mean(losses)
            self.logger.add_scalar("TrainingModelLoss", loss, self.step_i)
            wb.log({"TrainingModelLoss": loss}, step=self.step_i)

        # Print TaskName StepReward
        # ------------------------------
        for task_name in self.eval_tasks:
            step_reward = self.stats.get_recent_reward(task_name)
            # print(f"Step {self.step_i}\tReward: {step_reward}")
            self.logger.add_scalar("StepReward", step_reward, self.step_i)

        # Training agent
        # ------------------------------
        if self.step_i >= self.args.n_warm_up_steps and self.step_i % self.args.learning_freq == 0:
            self.agent.setup_normalizer(self.buffer.normalizer)
            for _ in range(self.args.n_policy_update_iters):
                states, actions, next_states, rewards = self.buffer.sample(self.args.batch_size, self.device)
                next_states_hat = self.model.sample(states, actions,
                                                    sampling_type=self.args.model_sampling_type)
                dones = self.is_done(next_states)

                self.agent.update(states, actions, rewards, next_states_hat, ~dones)

        # Evaluate Policy with Reset Environment
        # -------------------------------
        if self.args.eval_freq is not None and self.step_i % self.args.eval_freq == 0:
            self.last_avg_eval_score = self.evaluate_on_task()
            self.logger.add_scalar("EvaluateReturn", self.last_avg_eval_score, self.step_i)
            wb.log({"EvaluateEpReturn": self.last_avg_eval_score}, step=self.step_i)

        # Save agent parameters
        # ------------------------------
        if self.step_i >= self.args.n_warm_up_steps and self.step_i % self.args.save_freq == 0:
            os.makedirs(str(args.run_dir / 'incremental'), exist_ok=True)
            self.agent.save(str(args.run_dir / 'incremental' / ('model_step%i.pt' % self.step_i)))
            self.agent.save(str(self.args.run_dir / 'model.pt'))

        experiment_finished = self.step_i >= self.args.n_total_steps
        return DotMap(
            done=experiment_finished,
            step_i=self.step_i
        )

    def _train_model_epoch(self):
        losses = []
        for states, actions, state_deltas in self.buffer.train_batches(self.args.model_ensemble_size,
                                                                       self.args.model_batch_size):
            self.model_optimizer.zero_grad()
            loss = self.model.loss(states, actions, state_deltas)
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.args.model_grad_clip)
            self.model_optimizer.step()

        return losses

    def stop(self):
        self.env_loop.close()


if __name__ == "__main__":
    args = common_args()
    args = policy_function(args)
    args = value_function(args)
    args = dynamics_model(args)

    # Save Directory
    model_dir = Path('./models') / args.env_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in model_dir.iterdir()
                         if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    args.run_dir = model_dir / curr_run
    args.log_dir = args.run_dir / 'logs'
    os.makedirs(str(args.log_dir))
    logger = SummaryWriter(str(args.log_dir))
    # WandB Logger
    wb.init(project=f"chapter2_{args.env_name}",
            name=f"mage-vanilla-{args.seed}",
            config=args,
            entity="shaw7ock")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not args.use_cuda and args.n_training_threads is not None:
        torch.set_num_threads(args.n_training_threads)

    training = MainLoopTraining(logger, args)
    # MainLoop
    res = DotMap(done=False)
    while not res.done:
        res = training.train()

    training.stop()


