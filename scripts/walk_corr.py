"""Correlate commanded path velocity with measured torso velocity under the task's own command sampling."""
import sys, torch, numpy as np
sys.path.insert(0, "scripts")
from eval_policy import load_policy
task, ck = sys.argv[1], sys.argv[2]
play = len(sys.argv) > 3 and sys.argv[3] == "play"
env, wenv, policy = load_policy(task, ck, 256, play=play)
c = env.command_manager.get_term("ref"); r = env.scene["robot"]
obs = wenv.get_observations(); G, V = [], []
for i in range(400):
  with torch.no_grad(): obs, _, _, _ = wenv.step(policy(obs))
  yaw = c.pf_yaw; cs, sn = torch.cos(yaw), torch.sin(yaw)
  vw = r.data.root_link_lin_vel_w[:, :2]
  vp = torch.stack([cs * vw[:, 0] + sn * vw[:, 1], -sn * vw[:, 0] + cs * vw[:, 1]], -1)
  if i > 50:
    G.append(c.g[:, 4:6].cpu().numpy()); V.append(vp.cpu().numpy())
G, V = np.concatenate(G), np.concatenate(V)
for k, n in enumerate("xy"):
  print(f"v{n}: corr {np.corrcoef(G[:,k], V[:,k])[0,1]:.3f}, slope {np.polyfit(G[:,k], V[:,k], 1)[0]:.3f}, mean|cmd| {np.abs(G[:,k]).mean():.3f}")
