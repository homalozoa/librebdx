# Contributing

Issues and pull requests are welcome, especially:

- results on other GPUs or simulator versions;
- closer matches to the paper (see the deviations table in `docs/REPORT.md`, Sec. 4);
- fixes to anything that does not run as documented.

Guidelines:

- Keep the style of the surrounding code: 2-space indentation, short docstrings that give the usage line, no new heavy dependencies.
- When a change affects a number in `docs/REPORT.md`, update the number and say how it was measured.
- Do not commit training logs, checkpoints, renders or other files under `results/`; attach them to the pull request or link them.
- By contributing you agree that your contribution is licensed under the Apache License 2.0.
