# Publishing this project to GitHub

This folder is already structured as a GitHub repository. You do not need to upload files one by one.

## Option A — GitHub Desktop (easiest if you are new to Git)

1. Extract the repository ZIP to a permanent folder on your computer.
2. Open GitHub Desktop.
3. Choose **File -> Add local repository** and select the extracted folder.
4. If GitHub Desktop says the folder is not yet a Git repository, choose **Create a repository here**.
5. Use the repository name `graphrag-parliamentary-reasoning`.
6. Keep the repository public if you want recruiters to see it.
7. Make an initial commit with a message such as `Initial portfolio release`.
8. Click **Publish repository**.

Do not tick an option that adds another README, `.gitignore` or licence if GitHub Desktop offers one; those files are already included.

## Option B — command line

From inside the extracted folder:

```bash
git init
git add .
git commit -m "Initial portfolio release"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/graphrag-parliamentary-reasoning.git
git push -u origin main
```

Create the empty repository on GitHub before the `git remote add` step. Do not initialise the GitHub-side repository with a README because this folder already contains one.

## After publishing

On the GitHub repository page:

1. Add the description: `Graph-guided RAG vs hybrid RAG for multi-hop reasoning over UK parliamentary speech.`
2. Add topics such as `graphrag`, `rag`, `nlp`, `information-retrieval`, `knowledge-graph`, `python`, `llm`, `networkx`, `chromadb`.
3. Pin the repository on your profile.
4. Check that the two figures embedded near the top of the README render correctly.
5. Do not upload the full source corpus or local model/vector-store directories that are excluded by `.gitignore`.
