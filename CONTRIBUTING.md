# Contributing

Contributions should make the research question easier to answer. Keep a change small enough that its assumptions, evidence, and limits are clear.

Before choosing work, read the [task assignments](docs/tasks.md) and [team briefing](docs/team-briefing.html).

## Start from your fork

Fork `neuriv/rl-cache-validity` on GitHub, then clone your fork and add the project as `upstream`:

```sh
git clone https://github.com/YOUR-USERNAME/rl-cache-validity.git
cd rl-cache-validity
git remote add upstream https://github.com/neuriv/rl-cache-validity.git
git fetch upstream
git switch -c short-topic-name upstream/main
```

For later work, fetch upstream again and create a fresh branch from `upstream/main`. Keep your work on your fork; open a pull request from `YOUR-USERNAME:short-topic-name` into `neuriv/rl-cache-validity:main`.

## Set up and run the smoke check

Use Python 3.11 or newer. The harness uses small random Llama and Qwen2 configurations and does not download checkpoints.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/rl-cache-validity-smoke --architecture both --device cpu --output /tmp/rl-cache-validity-cpu.json
```

On a supported Mac, also try MPS and keep its numerical results separate from CPU results:

```sh
.venv/bin/rl-cache-validity-smoke --architecture both --device mps --output /tmp/rl-cache-validity-mps.json
```

Choose an output location outside the repository. Before proposing a change, inspect `git status` and the diff. Stage only the files that belong in the contribution, then commit and push your branch:

```sh
git status --short
git diff
git add path/to/changed-file
git commit -m "Describe the change"
git push -u origin short-topic-name
```

Open a pull request on GitHub with base `neuriv/rl-cache-validity:main` and compare `YOUR-USERNAME:short-topic-name`.

For documentation-only work, Python setup and smoke runs are not needed. For literature-only edits, GitHub's web editor is fine: edit the file in your fork, commit to a new branch, and open the same kind of pull request. Link the sources you checked and distinguish a paper's claim from what its experiment demonstrates.

## Explain the change

Write the pull request so a researcher outside the implementation team can follow it. In plain English, state the question, the proposed mechanism and why it could matter, what changed and what was held fixed, the evidence including null results, the uncertainty and limits, and the next decision the evidence supports. Include exact commands, configs, seeds, source revision, package versions, and hardware details needed to reproduce a result.

Do not add checkpoints, large generated dumps, private machine paths, or credentials. Keep raw runs outside Git. Small sanitized result tables and the commands/configuration needed to reproduce them are useful; ensure metadata contains no personal paths. If you use a coding agent, have it read this guide before changing code or research docs, and review its diff and claims yourself.
