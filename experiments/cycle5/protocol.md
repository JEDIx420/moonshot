# Cycle 5 CUDA baseline protocol

Use the existing 1,095,744-parameter Moonshot transformer, with all training and
capability inference on Burn CUDA. Smoke seed 42 compares initial CUDA logits to
the original CPU forward pass at absolute tolerance 0.005 and runs eight Adam
updates on a fixed synthetic batch, requiring finite decreasing loss and changed
weights.

Freeze evaluation seeds 4109, 4201, and 4303. Available Git history identifies
these as launcher protocol seeds and contains no earlier experiment code using
them or exact seed fields in prior committed experiment data. Each seed uses its
own initialization and its own ChaCha8 permutation of the 36 distinct prompts
in the existing tool-workflow-basic domain (four registers and values 1–9).
Partition into 24 train, six validation, and six test prompts in that order.
Splits are disjoint within each seed. The seeds reuse the same finite domain.

Train for 300 fixed Adam steps at learning rate 0.001, with teacher forcing and
a full batch of 24. Use the final model and greedy autoregressive exact match,
including EOS, on all three splits. Do not tune using these seeds. The gate is
test exact match of at least 0.85 on each seed, which requires six of six test
prompts per seed. This baseline measures generalization to unseen register-value
pairs in this task family.
