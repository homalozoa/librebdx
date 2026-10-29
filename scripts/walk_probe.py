"""Command a constant velocity and report how far the robot actually moves."""
import sys, torch
sys.path.insert(0, "scripts")
from eval_policy import load_policy
task, ck = sys.argv[1], sys.argv[2]
v = [float(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0.5, 0, 0]
env, wenv, policy = load_policy(task, ck, 16)
c = env.command_manager.get_term("ref"); r = env.scene["robot"]
obs = wenv.get_observations()
c.set_command(torch.tensor([[0, 0, 0, 0, *v]] * 16, device=env.device))
p0 = r.data.root_link_pos_w[:, :2].clone(); falls = 0
for i in range(250):
  with torch.no_grad(): obs, _, _, _ = wenv.step(policy(obs))
  falls += int(env.termination_manager.terminated.sum())
  if i == 49: p0 = r.data.root_link_pos_w[:, :2].clone()
d = (r.data.root_link_pos_w[:, :2] - p0).norm(dim=-1).mean().item()
print(f"{task}: cmd {v} -> mean speed {d / 4.0:.3f} m/s over 4 s, falls {falls}")
