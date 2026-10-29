"""mjlab training entry point with the Disney-character tasks registered.

Usage: python scripts/train.py Disney-BDX-Standing --env.scene.num-envs 8192 ...
"""

import sys

from disneyrepro.tasks import register_all

register_all()

from mjlab.scripts.train import main  # noqa: E402

if __name__ == "__main__":
  sys.exit(main())
