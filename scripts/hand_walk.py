import sys, torch
from disneyrepro.tasks import register_all
register_all()
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
cfg = load_env_cfg("Disney-BDX-Walking", play=True); cfg.scene.num_envs = 64
cfg.terminations.pop("fatal_contact"); cfg.terminations.pop("self_collision")
env = ManagerBasedRlEnv(cfg, device="cuda"); env.reset()
c = env.command_manager.get_term("ref"); r = env.scene["robot"]; at = env.action_manager.get_term("joints")
c.set_command(torch.tensor([[0, 0, 0, 0, 0.5, 0, 0]] * 64, device="cuda"))
K = float(sys.argv[1]); p0 = None
for i in range(150):
  q = r.data.joint_pos
  sp = c.ref_q + K * (c.ref_q - q) - 0.02 * K * r.data.joint_vel
  env.step((sp - at._offset) / at._scale)
  if i == 25: p0 = r.data.root_link_pos_w[:, :2].clone(); pf0 = c.pf_pos.clone()
  if i % 25 == 24 and p0 is not None:
    print(i, "torso disp", float((r.data.root_link_pos_w[:, :2] - p0).norm(dim=-1).mean()), "pf disp", float((c.pf_pos - pf0).norm(dim=-1).mean()), "z", float(r.data.root_link_pos_w[:, 2].mean()))
