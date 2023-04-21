import time
import jax
import jax.numpy  as jnp
import torch
from nanorl.types import LogDict, Transition
from pathlib import Path
from typing import Any, Callable

import dm_env
import tqdm
from nanorl import agent, replay, specs

from nanorl.infra import Experiment, utils


EnvFn = Callable[[], dm_env.Environment]
AgentFn = Callable[[dm_env.Environment], agent.Agent]
ReplayFn = Callable[[dm_env.Environment], replay.ReplayBuffer]
LoggerFn = Callable[[], Any]


def train_loop(
    experiment: Experiment,
    env_fn: EnvFn,
    agent_fn: AgentFn,
    replay_fn: ReplayFn,
    max_steps: int,
    warmstart_steps: int,
    log_interval: int,
    checkpoint_interval: int,
    resets: bool,
    reset_interval: int,
    tqdm_bar: bool,
) -> None:
    env = env_fn()
    agent = agent_fn(env)
    replay_buffer = replay_fn(env)

    spec = specs.EnvironmentSpec.make(env)
    timestep = env.reset()
    replay_buffer.insert(timestep, None)

    start_time = time.time()
    if hasattr(agent, "_device"):
        transitions = Transition(
            observation=torch.zeros(replay_buffer._batch_size, 24, device=agent._device),
            action=torch.zeros(replay_buffer._batch_size, 6, device=agent._device),
            reward=torch.zeros(replay_buffer._batch_size, device=agent._device),
            discount=torch.zeros(replay_buffer._batch_size, device=agent._device),
            next_observation=torch.zeros(replay_buffer._batch_size, 24, device=agent._device),
        )
    else:
        transitions = Transition(
            observation=jnp.zeros((replay_buffer._batch_size, 24)),
            action=jnp.zeros((replay_buffer._batch_size, 6)),
            reward=jnp.zeros((replay_buffer._batch_size,)),
            discount=jnp.zeros((replay_buffer._batch_size,)),
            next_observation=jnp.zeros((replay_buffer._batch_size, 24,)),
        )

    for i in tqdm.tqdm(range(1, max_steps + 1), disable=not tqdm_bar):
        if i < warmstart_steps:
            action = spec.sample_action(random_state=env.random_state)
        else:
            # _transitions = replay_buffer.sample()
            # c_transitions.observation[:] = torch.tensor(_transitions.observation)
            # c_transitions.action[:] = torch.tensor(_transitions.action)
            # c_transitions.reward[:] = torch.tensor(_transitions.reward)
            # c_transitions.discount[:] = torch.tensor(_transitions.discount)
            # c_transitions.next_observation[:] = torch.tensor(_transitions.next_observation)
            # # if hasattr(agent, "_device"):
            # #     transitions = Transition(
            # #         *[torch.tensor(x, dtype=torch.float32).to(agent._device, non_blocking=True) for x in transitions]
            # #     )
            # transitions = Transition(
            #     observation=c_transitions.observation.to(agent._device, non_blocking=True),
            #     action=c_transitions.action.to(agent._device, non_blocking=True),
            #     reward=c_transitions.reward.to(agent._device, non_blocking=True),
            #     discount=c_transitions.discount.to(agent._device, non_blocking=True),
            #     next_observation=c_transitions.next_observation.to(agent._device, non_blocking=True),
            # )

            # agent, action = agent.sample_actions(timestep.observation)
            pass

        timestep = env.step(action)
        replay_buffer.insert(timestep, action)

        if timestep.last():
            experiment.log(utils.prefix_dict("train", env.get_statistics()), step=i)
            timestep = env.reset()
            replay_buffer.insert(timestep, None)

        update_time = 0
        if i >= warmstart_steps:
            if replay_buffer.is_ready():
                if hasattr(agent, "_device"):
                    torch.cuda.synchronize()
                else:
                    (jax.device_put(0.) + 0).block_until_ready()
                _s_time =time.time()
                agent, metrics = agent.update(transitions)
                update_time = time.time() - _s_time
                if hasattr(agent, "_device"):
                    torch.cuda.synchronize()
                else:
                    (jax.device_put(0.) + 0).block_until_ready()
                if i % log_interval == 0:
                    experiment.log(utils.prefix_dict("train", metrics), step=i)

        if checkpoint_interval >= 0 and i % checkpoint_interval == 0:
            experiment.save_checkpoint(agent, step=i)

        if i % log_interval == 0:
            experiment.log(
                {
                    "train/fps": int(i / (time.time() - start_time)),
                    "train/update_time": update_time
                }, step=i)

        if resets and i % reset_interval == 0:
            agent = agent_fn(env)

    # Save final checkpoint and replay buffer.
    experiment.save_checkpoint(agent, step=max_steps, overwrite=True)
    utils.atomic_save(experiment.data_dir / "replay_buffer.pkl", replay_buffer.data)


def eval_loop(
    experiment: Experiment,
    env_fn: EnvFn,
    agent_fn: AgentFn,
    num_episodes: int,
    max_steps: int,
) -> None:
    env = env_fn()
    agent = agent_fn(env)

    last_checkpoint = None
    while True:
        # Wait for new checkpoint.
        checkpoint = experiment.latest_checkpoint()
        if checkpoint == last_checkpoint or checkpoint is None:
            time.sleep(10.0)
        else:
            # Restore checkpoint.
            agent = experiment.restore_checkpoint(agent)
            i = int(Path(checkpoint).stem.split("_")[-1])
            print(f"Evaluating checkpoint at iteration {i}")

            # Eval!
            for _ in range(num_episodes):
                timestep = env.reset()
                while not timestep.last():
                    timestep = env.step(agent.eval_actions(timestep.observation))

            # Log statistics.
            log_dict = utils.prefix_dict("eval", env.get_statistics())
            experiment.log(log_dict, step=i)

            # Maybe log video.
            experiment.log_video(env.latest_filename, step=i)

            print(f"Done evaluating checkpoint {i}")
            last_checkpoint = checkpoint

            # Exit if we've evaluated the last checkpoint.
            if i >= max_steps:
                print(f"Last checkpoint (iteration {i}) evaluated, exiting")
                break
