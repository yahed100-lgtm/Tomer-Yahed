# Results Summary

This document summarizes the expected meaning of the results produced by the project.

## Figure A — Online EWC vs SGD

The first experiment demonstrates catastrophic forgetting.

When the model is trained with regular SGD on a sequence of tasks, it learns the current task but its accuracy on older tasks decreases. This is the main phenomenon known as catastrophic forgetting.

Online EWC reduces this effect by adding a penalty term to the loss function. This penalty discourages large changes to parameters that were important for previous tasks, according to the Fisher information.

The expected result is that EWC keeps better performance on older tasks compared to plain SGD.

## Figure B — Multi-task comparison

The second experiment compares several approaches across multiple tasks.

A separate model for each task gives an upper-bound style comparison because each model only needs to specialize in one task.

A regular neural network with dropout still suffers from forgetting because dropout is not specifically designed to protect knowledge from old tasks.

EWC performs better in a continual-learning setting because it explicitly uses information from previous tasks to reduce harmful parameter changes.

## Figure C — Fisher overlap

The third experiment analyzes how similar or different tasks are inside the network.

The Fisher information measures how important each parameter is for a task. By comparing Fisher vectors between tasks, the project estimates how much overlap exists between the parameters used by different tasks.

Low permutation tasks are expected to have higher Fisher overlap because the images remain more similar to the original MNIST structure.

High permutation tasks are expected to have lower Fisher overlap, especially in earlier layers, because the input structure changes more strongly.

## Are the results fixed or calculated?

The results are calculated by the scripts during training and evaluation.

However, the project uses fixed random seeds. Because of this, repeated runs on the same environment should produce very similar results. This is intentional and improves reproducibility.

Therefore, similar results between runs should not be interpreted as hardcoded results. It means the experiment is deterministic or mostly deterministic under the same random seed and software environment.

## Scientific accuracy

The project reproduces the central ideas of the paper, but it should be described as an adapted reproduction rather than an exact one-to-one reproduction.

The main concepts are correctly represented:

- Catastrophic forgetting.
- Sequential task learning.
- Permuted MNIST.
- EWC regularization.
- Fisher information.
- Fisher overlap between tasks.

Some implementation choices, such as online EWC and the number of seeds, may differ from the original paper. These choices are acceptable as long as they are clearly explained.
