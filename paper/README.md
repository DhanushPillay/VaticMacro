# paper/

Academic artifacts kept for reviewers — **not** part of the app.

- `humanized_paper.tex`, `plagiarism_report.tex`, `references_list.tex`, `plagiarism_screenshot.png`, `VaticMacro_Details.pdf` live here.
- Excluded from Docker image via `.dockerignore` (`paper/`).
- Ignored by git except this `README.md` (see `.gitignore`).
- Do not import in `app/` or `src/`. Do not block deploys if missing.
