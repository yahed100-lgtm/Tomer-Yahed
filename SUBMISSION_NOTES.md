# Submission Notes

This repository contains a reproduction project for catastrophic forgetting and EWC.

The main code files were tested locally and are intentionally left unchanged before submission in order to avoid introducing new bugs close to the deadline.

The documentation files explain:

- What each experiment does.
- Why fixed random seeds are used.
- Why repeated runs may produce similar results.
- Why this should be considered an adapted reproduction rather than an exact pixel-perfect reproduction.

## Recommended final check before submission

Because the full experiments may take a long time, do not rerun everything close to the deadline unless necessary.

A safe final check is to verify that the repository includes:

- README.md
- RESULTS.md
- AI_DOCUMENTATION.md
- requirements.txt
- .gitignore
- The three main graph scripts: `main-graph A.py`, `main-graph_B.py`, and `main-GraphC.py`
- Output images/results generated from previous successful runs

## What to tell the examiner if asked

The code trains the models and calculates the results. The results are not manually typed into the code.

The reason the same or very similar result appears in repeated runs is that fixed random seeds are used for reproducibility.

This is common in machine-learning experiments because it allows the experiment to be checked consistently.
