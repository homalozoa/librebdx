"""Name -> GaitStyle factory for all authored styles."""


def get(name):
  if name in ("dino_gait", "injured_gait"):
    from disneyrepro.gait.styles import dino

    return getattr(dino, name)()
  raise KeyError(name)
