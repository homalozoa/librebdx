"""Registers all Disney-character tasks with mjlab (lazy, so importing is cheap)."""

_done = False


def _patch_only_positive_rewards():
  """Clip the total per-step reward at zero (legged_gym's only_positive_rewards).

  With the paper's penalty weights, early exploration noise makes the summed
  penalties exceed the survival bonus, so PPO otherwise learns to terminate
  as early as possible.
  """
  from mjlab.managers.reward_manager import RewardManager

  if getattr(RewardManager, "_disney_patched", False):
    return
  orig = RewardManager.compute

  import os

  clip = os.environ.get("DISNEY_CLIP_REWARD", "0") == "1"

  def compute(self, dt):
    r = orig(self, dt)
    if clip:
      r.clamp_(min=0.0)
    return r

  RewardManager.compute = compute
  RewardManager._disney_patched = True


def register_all():
  global _done
  if _done:
    return
  _patch_only_positive_rewards()
  from disneyrepro.bdx import tasks as bdx_tasks

  bdx_tasks.register()
  try:
    from disneyrepro.olaf import tasks as olaf_tasks

    olaf_tasks.register()
  except ImportError:
    pass
  _done = True
