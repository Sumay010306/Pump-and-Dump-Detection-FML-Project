# Publish this project to a Git repository

This folder is an independent local Git repository with a `main` branch. The prepared source files are staged for review. No commit, remote, or push has been made for you.

## Included in Git

- Dashboard and Python source modules
- Tests and pinned dependency lists
- README, setup scripts and executed analysis notebook
- Small real CSV examples and dashboard preview
- Data provenance manifests and small evaluation reports

## Kept out of Git

- Downloaded exchange/research archives
- Full cleaned candle dataset and prediction history
- Trained model binaries
- Generated standalone HTML report
- Virtual environments, caches and credentials

The excluded files can be regenerated with the documented pipeline. Use a release attachment or a separate file download for the complete project ZIP rather than committing it.

## First push

Open a terminal in this folder. Inspect the staged files:

```powershell
git status
git diff --cached --stat
```

Create the first commit:

```powershell
git commit -m "Initial PumpWatch project"
```

If Git asks for your identity, set it for this repository (use your own name and email):

```powershell
git config user.name "Your Name"
git config user.email "your-email@example.com"
```

Then run the commit command again.

Create an empty repository on your Git hosting service. Do not initialize that remote with a README, license or .gitignore because this project already has its own source history. Copy the remote URL, then run:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/pumpwatch.git
git push -u origin main
```

Replace the example URL with the real repository URL. Sign in using your Git host's supported credential flow; do not put tokens into source files or remote URLs. No credentials are included in this project.

## Future updates

```powershell
git status
git add .
git diff --cached --stat
git commit -m "Describe your changes"
git push
```

`.gitignore` protects only untracked files. If you later deliberately force-add a generated file, it will become tracked; avoid `git add -f` for datasets and models.

## If using the source ZIP

The source ZIP excludes Git's internal `.git` directory. After extracting it, initialize and stage it once before following the steps above:

```powershell
git init -b main
git add .
```
