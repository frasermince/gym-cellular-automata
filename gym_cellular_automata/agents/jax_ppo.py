# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpool_xla_jaxpy
import os
import random
import time
from dataclasses import dataclass
from typing import Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm
import json

from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter

# Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
# Fix CUDNN non-determinisim; https://github.com/google/jax/issues/4823#issuecomment-952835771
os.environ["TF_XLA_FLAGS"] = (
    "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
)
os.environ["TF_CUDNN DETERMINISTIC"] = "1"


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "firefighter"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Firefighter"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


class Network(nn.Module):
    log_grid_shapes: bool = False

    @nn.compact
    def __call__(self, grid, position):
        if self.log_grid_shapes:
            print(f"Initial grid shape: {grid.shape}")
            print(f"Initial position shape: {position.shape}")

        grid = grid[..., None]
        if self.log_grid_shapes:
            print(f"Grid shape after adding channel dim: {grid.shape}")

        grid = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(grid)
        if self.log_grid_shapes:
            print(f"Grid shape after first conv: {grid.shape}")

        grid = nn.relu(grid)
        grid = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(grid)
        if self.log_grid_shapes:
            print(f"Grid shape after second conv: {grid.shape}")

        grid = nn.relu(grid)
        grid = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(grid)
        if self.log_grid_shapes:
            print(f"Grid shape after third conv: {grid.shape}")

        grid = nn.relu(grid)

        grid_features = grid.reshape(grid.shape[0], -1)
        if self.log_grid_shapes:
            print(f"Grid features shape after flatten: {grid_features.shape}")

        grid_features = nn.Dense(256)(grid_features)
        if self.log_grid_shapes:
            print(f"Grid features shape after dense: {grid_features.shape}")

        grid_features = nn.relu(grid_features)

        # Process position input
        position_features = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(position)
        if self.log_grid_shapes:
            print(f"Position features shape after dense: {position_features.shape}")

        position_features = nn.relu(position_features)

        # Combine features
        combined = jnp.concatenate([grid_features, position_features], axis=-1)
        if self.log_grid_shapes:
            print(f"Combined features shape: {combined.shape}")

        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
            combined
        )
        if self.log_grid_shapes:
            print(f"Final features shape: {x.shape}")

        x = nn.relu(x)
        return x


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    action_dims: Sequence[int]

    @nn.compact
    def __call__(self, x):
        # Define shared network layers
        features = nn.Dense(64)(x)
        features = nn.relu(features)
        features = nn.Dense(64)(features)
        features = nn.relu(features)

        # Multiple heads for each action dimension
        logits = []

        for dim in self.action_dims:
            head = nn.Dense(dim)(features)
            logits.append(head)

        return logits


@flax.struct.dataclass
class AgentParams:
    network_params: flax.core.FrozenDict
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict


@flax.struct.dataclass
class Storage:
    grid_obs: jnp.array
    position_obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array


@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


def run_rollout_loop(env, num_iterations, num_envs=8):
    args = Args()
    args.batch_size = int(num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)

    # env setup
    # envs = envpool.make(
    #     args.env_id,
    #     env_type="gym",
    #     num_envs=num_envs,
    #     episodic_life=True,
    #     reward_clip=True,
    #     seed=args.seed,
    # )
    # envs.num_envs = num_envs
    # envs.action_space = envs.action_space
    # envs.observation_space = envs.observation_space
    # envs.is_vector_env = True
    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(num_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(num_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(num_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(num_envs, dtype=jnp.int32),
    )
    # handle, recv, send, step_env = envs.xla()

    def step_env_wrapped(episode_stats, action, obs, info):
        (
            next_obs,
            reward,
            next_done,
            truncated,
            next_info,
        ) = env.stateless_step(action, obs, info)
        new_episode_return = episode_stats.episode_returns + info["reward"]
        new_episode_length = episode_stats.episode_lengths + 1
        episode_stats = episode_stats.replace(
            episode_returns=(new_episode_return)
            * (1 - info["terminated"])
            * (1 - info["TimeLimit.truncated"]),
            episode_lengths=(new_episode_length)
            * (1 - info["terminated"])
            * (1 - info["TimeLimit.truncated"]),
            # only update the `returned_episode_returns` if the episode is done
            returned_episode_returns=jnp.where(
                info["terminated"] + info["TimeLimit.truncated"],
                new_episode_return,
                episode_stats.returned_episode_returns,
            ),
            returned_episode_lengths=jnp.where(
                info["terminated"] + info["TimeLimit.truncated"],
                new_episode_length,
                episode_stats.returned_episode_lengths,
            ),
        )
        return episode_stats, (
            next_obs,
            reward,
            next_done,
            next_info,
        )

    # assert isinstance(
    #     envs.action_space, gym.spaces.Discrete
    # ), "only discrete action space is supported"

    def linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches * args.update_epochs) gradient updates
        frac = (
            1.0
            - (count // (args.num_minibatches * args.update_epochs))
            / args.num_iterations
        )
        return args.learning_rate * frac

    network = Network()
    actor = Actor(action_dims=env.action_space.nvec[0])
    critic = Critic()

    grid_sample, (ca_params, position, t) = env.observation_space.sample()
    network_params = network.init(network_key, grid_sample, jnp.array(position))
    grid_sample, (ca_params, position, t) = env.observation_space.sample()
    actor_params = actor.init(
        actor_key,
        network.apply(
            network_params,
            grid_sample,
            jnp.array(position),
        ),
    )
    grid_sample, (ca_params, position, t) = env.observation_space.sample()
    critic_params = critic.init(
        critic_key,
        network.apply(
            network_params,
            grid_sample,
            jnp.array(position),
        ),
    )

    agent_state = TrainState.create(
        apply_fn=None,
        params=flax.core.freeze(
            {
                "network_params": network_params,
                "actor_params": actor_params,
                "critic_params": critic_params,
            }
        ),
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=linear_schedule if args.anneal_lr else args.learning_rate,
                eps=1e-5,
            ),
        ),
    )
    network.apply = jax.jit(network.apply)
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)

    # ALGO Logic: Storage setup
    grid, (ca_params, position, t) = env.observation_space
    storage = Storage(
        grid_obs=jnp.zeros((args.num_steps,) + grid.shape),
        position_obs=jnp.zeros((args.num_steps,) + position.shape),
        actions=jnp.zeros(
            (args.num_steps,) + env.action_space.shape,
            dtype=jnp.int32,
        ),
        logprobs=jnp.zeros(((args.num_steps,) + env.action_space.shape)),
        dones=jnp.zeros((args.num_steps, num_envs)),
        values=jnp.zeros((args.num_steps, num_envs)),
        advantages=jnp.zeros((args.num_steps, num_envs)),
        returns=jnp.zeros((args.num_steps, num_envs)),
        rewards=jnp.zeros((args.num_steps, num_envs)),
    )

    @jax.jit
    def get_action_and_value(
        agent_state: TrainState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
        step: int,
        key: jax.random.PRNGKey,
    ):
        """sample action, calculate value, logprob, entropy, and update storage"""
        next_grid_obs, (ca_params, next_position_obs, t) = next_obs
        hidden = network.apply(
            agent_state.params["network_params"],
            next_grid_obs,
            next_position_obs,
        )
        logits_set = actor.apply(agent_state.params["actor_params"], hidden)

        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
        logprobs = []
        actions = []
        for logits in logits_set:
            key, subkey = jax.random.split(key)
            u = jax.random.uniform(subkey, shape=logits.shape)
            action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
            logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
            logprobs.append(logprob)
            actions.append(action)
        logprobs = jnp.stack(logprobs, axis=-1)
        actions = jnp.stack(actions, axis=-1)

        value = critic.apply(agent_state.params["critic_params"], hidden)
        storage = storage.replace(
            grid_obs=storage.grid_obs.at[step].set(next_grid_obs),
            position_obs=storage.position_obs.at[step].set(next_position_obs),
            dones=storage.dones.at[step].set(next_done),
            actions=storage.actions.at[step].set(actions),
            logprobs=storage.logprobs.at[step].set(logprobs),
            values=storage.values.at[step].set(value.squeeze()),
        )
        return storage, actions, key

    @jax.jit
    def get_action_and_value2(
        params: flax.core.FrozenDict,
        x: tuple[np.ndarray, np.ndarray],
        action: np.ndarray,
    ):
        """calculate value, logprob of supplied `action`, and entropy"""
        grid, position = x
        hidden = network.apply(params["network_params"], grid, position)
        logits_set = actor.apply(params["actor_params"], hidden)
        # normalize the logits https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        entropies = []
        logprobs = []
        for i, logit in enumerate(logits_set):
            logprob = jax.nn.log_softmax(logit)[
                jnp.arange(action.shape[0]), action[:, i]
            ]
            logprobs.append(logprob)
            logits = logit - jax.scipy.special.logsumexp(logit, axis=-1, keepdims=True)
            logits = logits.clip(min=jnp.finfo(logits.dtype).min)
            p_log_p = logits * jax.nn.softmax(logits)
            entropy = -p_log_p.sum(-1)
            entropies.append(entropy)
        value = critic.apply(params["critic_params"], hidden).squeeze()
        return jnp.stack(logprobs, axis=-1), jnp.stack(entropies, axis=-1), value

    @jax.jit
    def compute_gae(
        agent_state: TrainState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
    ):
        storage = storage.replace(advantages=storage.advantages.at[:].set(0.0))

        grid, (ca_params, position, t) = env.observation_space.sample()
        next_value = critic.apply(
            agent_state.params["critic_params"],
            network.apply(
                agent_state.params["network_params"],
                grid,
                position,
            ),
        ).squeeze()
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - storage.dones[t + 1]
                nextvalues = storage.values[t + 1]
            delta = (
                storage.rewards[t]
                + args.gamma * nextvalues * nextnonterminal
                - storage.values[t]
            )
            lastgaelam = (
                delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            )
            storage = storage.replace(
                advantages=storage.advantages.at[t].set(lastgaelam)
            )
        storage = storage.replace(returns=storage.advantages + storage.values)
        return storage

    @jax.jit
    def update_ppo(
        agent_state: TrainState,
        storage: Storage,
        key: jax.random.PRNGKey,
    ):
        b_grid_obs = storage.grid_obs.reshape(
            (-1,) + env.observation_space[0].shape[1:]
        )
        b_position_obs = storage.position_obs.reshape(
            (-1,) + env.observation_space[1][1].shape[1:]
        )
        b_logprobs = storage.logprobs.reshape((-1,) + (env.action_space.shape[-1],))
        b_actions = storage.actions.reshape((-1,) + (env.action_space.shape[-1],))
        b_advantages = storage.advantages.reshape(-1)
        b_returns = storage.returns.reshape(-1)

        def ppo_loss(params, x, a, logp, mb_advantages, mb_returns):
            newlogprob, entropy, newvalue = get_action_and_value2(params, x, a)
            logratio = newlogprob - logp
            ratio = jnp.exp(logratio)
            approx_kl = ((ratio - 1) - logratio).mean()

            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

            mb_advantages = mb_advantages[:, None, None]

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * jnp.clip(
                ratio, 1 - args.clip_coef, 1 + args.clip_coef
            )
            # TODO: I'm very suspicious of this line. We are meaning the two
            # logprobs for the actions together. Soon as we add extensions this
            # will be a third action. We need to confirm this is right.
            pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

            # Value loss
            v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
            return loss, (
                pg_loss,
                v_loss,
                entropy_loss,
                jax.lax.stop_gradient(approx_kl),
            )

        ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
        for _ in range(args.update_epochs):
            key, subkey = jax.random.split(key)
            b_inds = jax.random.permutation(subkey, args.batch_size, independent=True)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = (
                    ppo_loss_grad_fn(
                        agent_state.params,
                        (b_grid_obs[mb_inds], b_position_obs[mb_inds]),
                        b_actions[mb_inds],
                        b_logprobs[mb_inds],
                        b_advantages[mb_inds],
                        b_returns[mb_inds],
                    )
                )
                agent_state = agent_state.apply_gradients(grads=grads)
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, report = env.reset()
    next_done = jnp.full(num_envs, False)
    next_info = {
        "TimeLimit.truncated": jnp.full(num_envs, False),
        "terminated": jnp.full(num_envs, False),
        "steps_elapsed": jnp.zeros(num_envs),
        "reward_accumulated": jnp.zeros(num_envs),
        "reward": jnp.zeros(num_envs),
    }

    @jax.jit
    def rollout(
        agent_state,
        episode_stats,
        next_obs,
        next_done,
        next_info,
        storage,
        key,
        global_step,
    ):
        # Remove @jax.jit and use jax.lax.fori_loop instead

        def body_fun(step, carry):
            (
                agent_state,
                episode_stats,
                next_obs,
                next_done,
                next_info,
                storage,
                key,
                global_step,
            ) = carry

            global_step += num_envs
            storage, action, key = get_action_and_value(
                agent_state, next_obs, next_done, storage, step, key
            )

            episode_stats, (next_obs, reward, next_done, next_info) = step_env_wrapped(
                episode_stats, action, next_obs, next_info
            )
            storage = storage.replace(rewards=storage.rewards.at[step].set(reward))

            return (
                agent_state,
                episode_stats,
                next_obs,
                next_done,
                next_info,
                storage,
                key,
                global_step,
            )

        init_carry = (
            agent_state,
            episode_stats,
            next_obs,
            next_done,
            next_info,
            storage,
            key,
            global_step,
        )
        return jax.lax.fori_loop(0, args.num_steps, body_fun, init_carry)

    for iteration in tqdm(range(1, num_iterations + 1), desc="Training"):
        iteration_time_start = time.time()
        (
            agent_state,
            episode_stats,
            next_obs,
            next_done,
            next_info,
            storage,
            key,
            global_step,
        ) = rollout(
            agent_state,
            episode_stats,
            next_obs,
            next_done,
            next_info,
            storage,
            key,
            global_step,
        )
        storage = compute_gae(agent_state, next_obs, next_done, storage)
        agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key = update_ppo(
            agent_state,
            storage,
            key,
        )
        avg_episodic_return = np.mean(
            jax.device_get(episode_stats.returned_episode_returns)
        )
        current_episodic_return = np.mean(jax.device_get(episode_stats.episode_returns))
        print(
            f"global_step={global_step}, avg_episodic_return={avg_episodic_return}, current_episodic_return={current_episodic_return}"
        )

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/avg_episodic_return", avg_episodic_return, global_step
        )
        writer.add_scalar(
            "charts/avg_episodic_length",
            np.mean(jax.device_get(episode_stats.returned_episode_lengths)),
            global_step,
        )
        writer.add_scalar(
            "charts/learning_rate",
            agent_state.opt_state[1].hyperparams["learning_rate"].item(),
            global_step,
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/loss", loss.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )
        writer.add_scalar(
            "charts/SPS_update",
            int(num_envs * args.num_steps / (time.time() - iteration_time_start)),
            global_step,
        )

    # Save grid observations to JSON file

    grid_obs_list = jax.device_get(storage.grid_obs).tolist()
    with open(f"runs/{run_name}_grid_obs.json", "w") as f:
        json.dump(grid_obs_list, f)
    # envs.close()
    writer.close()
