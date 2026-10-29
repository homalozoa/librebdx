import sys, torch
sys.path.insert(0, "scripts")
from eval_policy import load_policy
env, wenv, policy = load_policy("Disney-BDX-Episodic-jump", sys.argv[1], 1)
obs = wenv.get_observations(); c = env.command_manager.get_term("ref"); r = env.scene["robot"]
s = env.scene.sensors["feet_contact"]
for i in range(int(1.8 / env.step_dt)):
  with torch.no_grad(): obs, _, _, _ = wenv.step(policy(obs))
  t = float(c.t[0])
  if 0.8 < t < 1.6:
    print(f"t={t:.2f} contact={(s.data.found[0] > 0).int().tolist()} ref_c={c.ref_contact[0].int().tolist()} z={float(r.data.root_link_pos_w[0,2]):.3f} ref_z={float(c.ref_pos[0,2]):.3f}")
