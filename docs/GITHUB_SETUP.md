# Maintaining the published ParliamentAI repository

The public repository is already published at **[RonitBulani/graphrag-uk-parliament](https://github.com/RonitBulani/graphrag-uk-parliament)**. Do **not** initialise a new repository or follow an initial-publication workflow for the existing project.

## Clone or update safely

To clone onto a new computer:

```powershell
git clone https://github.com/RonitBulani/graphrag-uk-parliament.git
cd graphrag-uk-parliament
```

To inspect an existing local clone before making changes:

```powershell
git status -sb
git remote -v
git branch --show-current
```

If you have uncommitted changes, review and preserve them before switching branches or pulling. Do not run destructive reset, clean or forced-push commands as a shortcut.

## Propose updates through a pull request

Start with a clean working tree and an up-to-date `main` branch:

```powershell
git switch main
git pull --ff-only origin main
git switch -c docs/describe-your-update
```

Edit the relevant files, review `git diff`, and run appropriate checks. Then commit and push your branch:

```powershell
git add path/to/changed-file.md
git commit -m "docs: describe the update"
git push -u origin docs/describe-your-update
```

Open a pull request targeting `main`. Inspect the file diff and GitHub Actions results before merging. Replace the example branch and path names with your actual changes.

## Repository hygiene

- Keep `data/vectorstore/`, generated graph files, `archive.zip`, model caches, credentials and local-only data out of commits (see [`.gitignore`](../.gitignore)).
- Separate documentation changes from application or research-method changes where possible.
- Keep claims about the **original 50-question benchmark** distinct from verification of the **current interactive application**.
- Do not publish a public demo URL, video recording or reproducibility claim unless the corresponding asset or workflow has actually been verified.
- Consult [README](../README.md), [Architecture](ARCHITECTURE.md), [Reproducibility](REPRODUCIBILITY.md) and [Docker setup](../DOCKER_SETUP.md) before preparing a release.
