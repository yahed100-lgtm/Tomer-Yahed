# Catastrophic Forgetting and Elastic Weight Consolidation (EWC)

This project reproduces and analyzes central ideas from the paper **"Overcoming Catastrophic Forgetting in Neural Networks"** by Kirkpatrick et al.

The goal of the project is to demonstrate the phenomenon of **catastrophic forgetting** in neural networks and to show how **Elastic Weight Consolidation (EWC)** can reduce this forgetting by protecting parameters that are important for previously learned tasks.

## What the project does

The project contains three main experiments:

### Figure A — Online EWC vs SGD

This experiment trains a neural network on a sequence of permuted MNIST tasks.

The goal is to compare:

- **SGD without protection** — learns the current task but tends to forget previous tasks.
- **Online EWC** — adds a regularization penalty that discourages changes to parameters that were important for previous tasks.

The expected result is that SGD suffers more from forgetting, while EWC keeps higher accuracy on earlier tasks.

### Figure B — Many-task comparison

This experiment compares performance over multiple permuted MNIST tasks.

It compares:

- A separate model trained independently for each task.
- A regular neural network with dropout.
- A network trained using EWC.

The purpose is to show that EWC provides better continual-learning behavior than regular training when the same model must learn several tasks sequentially.

### Figure C — Fisher overlap between tasks

This experiment analyzes the relationship between tasks using the Fisher information.

The project compares tasks with different levels of permutation:

- Low permutation — small local permutation in the image.
- High permutation — stronger/global permutation.

The Fisher overlap is measured across layers in the network. Higher overlap means that two tasks rely on more similar parameters, while lower overlap means that the tasks use the network differently.

## Important note about reproducibility

The experiments use fixed random seeds. This means that the results are expected to be very similar between runs.

This does **not** mean the results are hardcoded. The scripts train models, calculate accuracies/Fisher values, and generate the graphs again. The fixed seeds are used so the results are reproducible and can be checked consistently.

## How to run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run each experiment directly from its Python script:

```bash
python "main-graph A.py"
python "main-graph_B.py"
python "main-GraphC.py"
```

If your local filenames are slightly different, use the exact filenames that appear in your repository.

It is recommended to run one script at a time because each experiment can take a long time.

## Expected outputs

The scripts generate graph image files showing the results of the experiments.

Because these are neural-network training experiments, the full run can take a long time depending on the computer, CPU/GPU, and installed PyTorch version.

## Project interpretation

This project should be understood as a **qualitative and adapted reproduction** of the original paper's experiments.

The purpose is not to copy the original results pixel-by-pixel, but to reproduce the main scientific idea:

1. Neural networks forget old tasks when trained sequentially.
2. EWC reduces forgetting by protecting important weights.
3. Fisher information can be used to analyze which parameters are important for different tasks.

## Repository notes

The source code files were tested locally before submission. The repository also includes documentation and result files to make the project easier to understand and evaluate.
