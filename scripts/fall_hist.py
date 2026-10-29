import sys, torch, numpy as np
sys.path.insert(0, "scripts")
from eval_policy import load_policy
env, wenv, policy = load_policy(sys.argv[1], sys.argv[2], 1024, play=False)
obs = wenv.get_observations(); ages = []
for i in range(500):
  age = env.episode_length_buf.clone()
  with torch.no_grad(): obs, _, _, _ = wenv.step(policy(obs))
  t = env.termination_manager.terminated
  if i > 60:
    ages += (age[t] * 0.02).tolist()
a = np.array(ages)
print("falls", len(a), "per env-s", len(a) / (1024 * 8.8))
for lo, hi in ((0, 0.5), (0.5, 1), (1, 2), (2, 5), (5, 100)):
  print(f"episode age [{lo},{hi}) s: {((a >= lo) & (a < hi)).sum()}")
