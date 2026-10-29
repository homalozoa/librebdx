import sys, numpy as np, torch
sys.path.insert(0, "scripts")
from eval_policy import load_policy
env, wenv, policy = load_policy(sys.argv[1], sys.argv[2], 512)
c = env.command_manager.get_term("ref")
obs = wenv.get_observations()
falls, allg = [], []
for i in range(1000):
  with torch.no_grad():
    obs, _, _, _ = wenv.step(policy(obs))
  allg.append(c.g.clone().cpu().numpy())
  term = env.termination_manager.terminated
  if term.any():
    # g already resampled after reset; use previous step's command
    falls.append(allg[-2][term.cpu().numpy()] if len(allg) > 1 else allg[-1][term.cpu().numpy()])
F = np.concatenate(falls) if falls else np.zeros((0, 7))
G = np.concatenate(allg)
print("falls", len(F), "per env-second", len(F) / (512 * 20))
names = ["dh", "hy", "hp", "hr", "vx", "vy", "wz"]
for j in range(7):
  print(f"{names[j]:3s} fall |mean| {np.abs(F[:, j]).mean():.3f}  overall |mean| {np.abs(G[:, j]).mean():.3f}")
for lo, hi in ((0, 0.2), (0.2, 0.5), (0.5, 0.8)):
  m = (np.abs(G[:, 4]) >= lo) & (np.abs(G[:, 4]) < hi); mf = (np.abs(F[:, 4]) >= lo) & (np.abs(F[:, 4]) < hi)
  print(f"|vx| in [{lo},{hi}): falls {mf.sum()} / time share {m.mean():.2f}")
for lo, hi in ((0, 0.3), (0.3, 0.7), (0.7, 1.1)):
  m = (np.abs(G[:, 6]) >= lo) & (np.abs(G[:, 6]) < hi); mf = (np.abs(F[:, 6]) >= lo) & (np.abs(F[:, 6]) < hi)
  print(f"|wz| in [{lo},{hi}): falls {mf.sum()} / time share {m.mean():.2f}")
