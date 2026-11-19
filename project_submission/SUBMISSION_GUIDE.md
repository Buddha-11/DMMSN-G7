# QA-DR-MADDPG Project Submission

This package contains the implementation, experiment evidence, report, and
editable LaTeX project for the queue-aware domain-randomized MADDPG study.

## Directory structure

- `code/src/vec/` - active simulator, queue, energy, safety, learning,
  training, and evaluation implementation.
- `code/scripts/` - command-line entry points for training, evaluation, the
  complete study, and the domain-randomized queue-weight ablation.
- `code/tests/` - automated tests for the active VEC implementation.
- `documentation/` - project README files, implementation checklist, and
  Python requirements.
- `report/QA_DR_MADDPG_Report.pdf` - compiled final report.
- `latex/` - self-contained Overleaf project. Its main document is
  `latex/main.tex`, and it compiles with pdfLaTeX.
- `results/full_study/` - full comparison logs, aggregate metrics,
  confidence intervals, figures, scenario manifest, and per-seed training
  metadata.
- `results/domain_randomization_ablation/` - direct domain-randomized
  queue-weight ablation logs, metrics, figures, and training metadata.

## Important files to inspect

- `code/src/vec/env.py` - end-to-end causal VEC simulator.
- `code/src/vec/learning/agent.py` - hybrid MADDPG actor-critic learner.
- `code/src/vec/training/trainer.py` - training, validation, checkpointing,
  and domain-randomization loop.
- `code/scripts/run_vec_study.py` - complete multi-seed study entry point.
- `code/scripts/run_domain_randomized_queue_ablation.py` - focused ablation
  entry point.
- `results/full_study/study_run_metrics.csv` - run-level study measurements.
- `results/full_study/learner_seed_confidence_intervals.csv` - uncertainty
  estimates across independently trained learner seeds.
- `results/full_study/logs/` - training and evaluation console logs.

## Reproduction environment

Install the dependencies from `documentation/requirements.txt`. Run commands
from the repository root so that the `src` package can be imported. The exact
study configuration, scenario split, and provenance records are stored with
each result set.

## Deliberately excluded large artifacts

PyTorch checkpoint and resume-state files, Python caches, temporary build
files, smoke-test outputs, and per-slot raw evaluation directories are not
duplicated in this submission package. The original workspace retains those
artifacts. Excluding them reduces the package by roughly 20 GB without
removing source code, logs, learning histories, configurations, aggregate
metrics, statistical comparisons, figures, or the report.
