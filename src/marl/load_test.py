import pickle
from pathlib import Path

import numpy as np
import ray
from ray.rllib import SampleBatch, Policy
from ray.rllib.algorithms import Algorithm
from ray.tune import register_env
from ray.tune import Tuner
from ray.tune.logger import pretty_print

from src.marl.common_marl import setup
from src.marl.corridor_env import CorridorEnv
from src.marl.centralised_critic import CentralizedCritic, CentralizedCriticModel
from gymnasium.spaces import Dict
import torch

from src.utils.scenarios import AsymmetricalTwoSlotCorridor
from src.utils.utils import Utils


def exec(ep_num):
  setup()

  policy: Policy = Policy.from_checkpoint(f"/home/toby/projects/uni/internship/Hybrid_LLM_MARL/test/episode_{ep_num}")

  # algo = Algorithm.from_checkpoint(
  #     "/home/toby/projects/uni/internship/Hybrid_LLM_MARL/output/CentralizedCritic_2024-08-21_12-09-09/CentralizedCritic_corridor_c9bf8_00000_0_2024-08-21_12-09-10/checkpoint_000000",
  #     policy_ids = ["pol1"],
  #     policy_mapping_fn = lambda agent_id, episode, worker, **kwargs: "pol1"
  # )

  env = CorridorEnv({
    "csv_path":            "/home/toby/projects/uni/internship/Hybrid_LLM_MARL/test",
    "csv_filename":        "tes1",
    "episode_step_limit":  50,
    "env_change_rate_eps": 0,  # 0 for no env change
    "scenario_name":       "Asymmetrical_Two_Slot_Corridor",
    "worker_index":        1
  })

  done = {'__all__': False}
  total_reward = 0
  observations = env.reset()[0]
  # print(observations)
  steps = 0
  ids = ["alice", "bob"]
  moves = []

  while not done["__all__"]:
    # action = algo.compute_actions(observations, policy_id = "pol1")
    action = {}
    for i in ids:
      action[i] = policy.compute_single_action(observations[i])[0]
    print(f"action: {action}")
    observations, reward, done, trunc, info = env.step(action)
    act_pair = {}
    try:
      if info["alice"]["valid_move"]:
        act_pair["alice"] = action["alice"]
      else:
        act_pair["alice"] = 0
      if info["bob"]["valid_move"]:
        act_pair["bob"] = action["bob"]
      else:
        act_pair["bob"] = 0
    except KeyError:
      pass
    moves.append(act_pair)
    steps += 1
    print(f"observations: {observations}")
    print(f"reward: {reward}")
    print(f"done: {done}")
    print(f"trunc: {trunc}")
    print(f"info: {info}")
    total_reward += sum(reward.values())
    print(f"total_reward {total_reward}")
    print(f"steps: {steps}")
    # input("Any key for next step")

  loc_list = [(env.agent_pos[i], env.agent_goal_pos[i], env.agent_starting_pos[i]) for i in env._agent_ids]
  avg_perf = Utils.calc_multiagent_avg_perf(loc_list)
  print(avg_perf)
  print(moves)
  return avg_perf


if __name__ == "__main__":
  ray.init()

  data = {}

  for i in range(1, 81):
    ep_num = i * 10
    data[ep_num] = []
    for _ in range(10):
      perf = exec(ep_num)
      data[ep_num].append(perf)

  ray.shutdown()

  for k, v in data.items():
    print(f"Episode: {k}: {np.mean(v)}")
