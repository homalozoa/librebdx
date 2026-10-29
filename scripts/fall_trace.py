import sys, torch, math
sys.path.insert(0, "scripts")
from eval_policy import load_policy
from mjlab.utils.lab_api.math import matrix_from_quat
env, wenv, policy = load_policy("Disney-BDX-Walking", sys.argv[1], 64)
c = env.command_manager.get_term("ref"); r = env.scene["robot"]
v = [float(x) for x in sys.argv[2].split(",")]
c.set_command(torch.tensor([[0, 0, 0, 0, *v]] * 64, device=env.device))
obs = wenv.get_observations(); first_fall = torch.full((64,), -1.0, device=env.device)
for i in range(400):
  with torch.no_grad(): obs, _, _, _ = wenv.step(policy(obs))
  t = env.termination_manager.terminated
  first_fall = torch.where(t & (first_fall < 0), torch.full_like(first_fall, i * 0.02), first_fall)
  if i % 25 == 0:
    _, R = c.torso_in_path_frame()
    yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0]); roll = torch.atan2(R[:, 2, 1], R[:, 2, 2]); pitch = torch.asin(-R[:, 2, 0].clamp(-1, 1))
    p, _ = c.torso_in_path_frame()
    print(f"t={i*0.02:.1f} |yaw| {yaw.abs().mean():.2f} |roll| {roll.abs().mean():.2f} |pitch| {pitch.abs().mean():.2f} pos err {p.norm(dim=-1).mean():.3f} falls so far {(first_fall>=0).sum().item()}")
f = first_fall[first_fall >= 0]
print("envs fallen", len(f), "median first fall time", f.median().item() if len(f) else None)
