# Titanic classification: Codex working instructions

## Scope

Maintain the notebook-based classification workflow and the traceability of its final model.

The user's current request takes precedence. This package authorizes no project changes by itself. Preserve the existing analytical intent and do not invent metrics, source data, successful tests, or model results.

## Repository baseline

Repository: JudeMirac/titanic
Inspected branch: main
Inspected commit: 3e06a54b19388d7da8375b8b99c9ae49433e7574
This is a remote source assessment, not proof of a clean local worktree. Recheck branch, current HEAD, existing changes, and any nested instructions before implementation.

- Workflow/Notebooks/01_EDA.ipynb through 06_Final_Model.ipynb form the six-stage workflow.
- Actual filenames include 02_Feature_engineer_testing.ipynb, 03_Hypothesis_Testing.ipynb, 04_Modeling.ipynb, and 05_Model_Evaluation.ipynb.
- Workflow/Data/ stores raw and derived inputs; Workflow/Final_Model.pkl and final_model_metadata.json are tracked artifacts.
- README documents a final logistic regression and threshold; its proposed Scripts/train_final_model.py path is absent from the inspected tree.

## Multi-agent workflow

Keep the lead focused on planning, integration, ambiguous decisions, and final review.

- titanic_explorer: Read-only repository mapping, dependency tracing, data-flow inspection, and targeted investigation. Model: gpt-6-luna; reasoning: high.
- titanic_worker: Small, clearly scoped code, notebook, documentation, and configuration changes. Model: gpt-6-luna; reasoning: high.
- titanic_engineer: Complex implementation, cross-file debugging, data-pipeline failures, and integration decisions. Model: gpt-6.1-sol; reasoning: medium.
- titanic_reviewer: Independent review for correctness, regressions, data leakage, reproducibility, and missed requirements. Model: gpt-6.1-sol; reasoning: medium.

Use one or two specialists for ordinary work. Delegate only bounded work with a clear output. Parallelize independent tasks and never give concurrent writers overlapping files. Ask the engineer to handle hard failures rather than repeatedly widening a routine worker's scope. Use independent review for meaningful analytical or functional changes. Do tiny tasks directly.

Read the relevant source before editing. Preserve unrelated user changes and established names. Review the final diff, perform proportionate verification, fix introduced regressions, and report changed paths, checks actually run, and limitations. Do not run expensive training, notebook execution, dataset regeneration, downloads, or external services merely for package validation.

## Project-specific rules

- Use actual tree paths rather than assuming the README's planned structure is implemented.
- Preserve split boundaries, preprocessing fit scope, and reproducibility when changing feature selection or hypothesis tests.
- Treat the documented logistic-regression settings and threshold 0.45 as existing design choices to verify, not guaranteed optimal values.
- Do not silently retrain or replace Final_Model.pkl or metadata; compare model, threshold, and metrics with reproducible evidence.
- Do not unpickle model artifacts merely to inspect the repository. Avoid rewriting notebook outputs unnecessarily.

## Verification

- Parse notebook JSON and validate with nbformat if available without cell execution.
- Parse final_model_metadata.json as JSON and inspect references to model and input paths.
- No executable training script, dependency manifest, or automated test suite was identified in the inspected tree.
- Execute notebooks only within an authorized modeling task with known dependencies, data, and output paths.

Do not install dependencies or regenerate artifacts for documentation-only edits. For code changes, choose the smallest relevant check and distinguish syntax validation from runtime or analytical validation. Preserve train/test boundaries where applicable. Stop broadening checks once the relevant risks are covered.

## Assessment limitations

- No notebook execution, model deserialization, or metric reproduction was performed.

Do not infer a deployment from another repository. Major redesign, destructive data changes, repository visibility changes, and deployment-architecture changes require explicit user authorization unless already part of the current request.
