"""Roll out a trained policy: tracking metrics + optional video of env 0.

python scripts/eval_policy.py TASK CHECKPOINT [--steps N] [--envs N] [--video out.mp4]
"""

import argparse
import os
from dataclasses import asdict

import numpy as np
import torch

from disneyrepro.tasks import register_all

register_all()
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402


def load_policy(task, ckpt, num_envs, render=False, play=True, cfg_hook=None):
  env_cfg = load_env_cfg(task, play=play)
  env_cfg.scene.num_envs = num_envs
  if cfg_hook:
    cfg_hook(env_cfg)
  if render and os.environ.get("DISNEY_RES"):  # e.g. DISNEY_RES=1280x720
    env_cfg.viewer.width, env_cfg.viewer.height = map(int, os.environ["DISNEY_RES"].split("x"))
  agent_cfg = load_rl_cfg(task)
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda", render_mode="rgb_array" if render else None)
  wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(wenv, asdict(agent_cfg), device="cuda")
  runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda")
  return env, wenv, runner.get_inference_policy(device="cuda")


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("task")
  ap.add_argument("ckpt")
  ap.add_argument("--steps", type=int, default=500)
  ap.add_argument("--envs", type=int, default=64)
  ap.add_argument("--video", default=None)
  a = ap.parse_args()
  env, wenv, policy = load_policy(a.task, a.ckpt, a.envs, render=a.video is not None)
  obs = wenv.get_observations()
  c = env.command_manager.get_term("ref")
  r = env.scene["robot"]
  frames, errs, falls = [], [], 0
  for i in range(a.steps):
    with torch.no_grad():
      act = policy(obs)
    obs, _, dones, _ = wenv.step(act)
    falls += int(env.termination_manager.terminated.sum())
    errs.append((r.data.joint_pos - c.ref_q).abs().mean(0).cpu().numpy())
    if a.video and i % 1 == 0:
      frames.append(env.render())
  errs = np.array(errs)
  print(f"falls {falls} over {a.steps} steps x {a.envs} envs; joint MAE {errs.mean():.4f} rad")
  print("per-joint MAE", np.round(errs.mean(0), 3).tolist())
  if a.video:
    import mediapy

    mediapy.write_video(a.video, np.asarray(frames), fps=50)
