import sys, torch, numpy as np, mediapy
sys.path.insert(0, "scripts")
from eval_policy import load_policy
env, wenv, policy = load_policy("Disney-BDX-Walking", sys.argv[1], 1, render=True)
c = env.command_manager.get_term("ref"); r = env.scene["robot"]
obs = wenv.get_observations(); frames = []
c.set_command(torch.tensor([[0, 0, 0, 0, 0.5, 0, 0]], device=env.device))
for i in range(200):
  with torch.no_grad(): obs, _, _, _ = wenv.step(policy(obs))
  frames.append(env.render())
  if i % 20 == 0:
    print(i, "phi", round(float(c.phi[0]), 2), "contact ref", c.ref_contact[0].tolist(), "found", (env.scene.sensors["feet_contact"].data.found[0] > 0).int().tolist(), "q err", round(float((r.data.joint_pos - c.ref_q).abs().mean()), 3))
mediapy.write_video(sys.argv[2], np.asarray(frames), fps=50)
