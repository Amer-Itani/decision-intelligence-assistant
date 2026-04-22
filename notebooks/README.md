# Notebooks

This folder contains the project notebook work for the Decision Intelligence
Assistant.

Planned notebook flow:

1. Load the Twitter support dataset.
2. Build a representative 10k-row sample.
3. Define and document the weak-labeling rule for priority.
4. Compare three feature setups:
   - TF-IDF only
   - engineered features only
   - TF-IDF + engineered features
5. Evaluate candidate classifiers.
6. Export the chosen model artifacts for backend inference.

Keep the notebook honest and review-ready:

- document every labeling choice
- use reproducible splits with `random_state`
- compare against simple baselines
- show both successes and failure cases
- avoid reporting any metric that was not actually produced by code
