# Publishing this repository to GitHub

The project code, documentation, tests, metrics and reference videos are all
ready in this folder. Choose one of the three options below to publish it.

## Option A — Push with Git (HTTPS + Personal Access Token)

1. Create an **empty** repository on GitHub
   (https://github.com/new) — do **not** tick "Add a README file",
   "Add .gitignore" or "Choose a license", so the repository stays empty.
2. Copy the repository URL, e.g.
   `https://github.com/<your-username>/NavRL6-MultiRobot-Navigation.git`
3. Inside this folder, run:

```bash
git init -b main
git add -A
git commit -m "NavRL6: six-robot RL navigation (static/dynamic/mixed), PPO actor-critic, metrics, tests"
git remote add origin https://github.com/<your-username>/NavRL6-MultiRobot-Navigation.git
git push -u origin main
```

When prompted for the password, paste a GitHub **Personal Access Token**
(Developer settings → Tokens → `repo` scope) instead of your account password.

## Option B — GitHub CLI (`gh`)

```bash
gh auth login
gh repo create NavRL6-MultiRobot-Navigation --public --source=. --push
```

Use `--private` instead of `--public` if the repository should be restricted.

## Option C — Upload from the web UI

1. Create an empty repository on GitHub.
2. Click **Add file → Upload files**, then drag the contents of this folder
   (the `navrl6/`, `tests/`, `results/`, `videos/` directories and
   `README.md`, `requirements.txt`, `.gitignore`) into the upload area.
3. Commit directly to `main`.

## Repository contents checklist

- [x] Package source code: `navrl6/`
- [x] Explanation: `README.md` (scenarios, reward table, metrics, NN + hyper-parameters)
- [x] Tests: `tests/` (37 unit tests, run with `python -m pytest tests/ -v`)
- [x] Metrics and figures: `results/`
- [x] Reference videos: `videos/`
