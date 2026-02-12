# AI Phantom

## Main Experiment Pipeline

1. Trains a DQN agent in the maze environment using configuration files and saves checkpoints.

```
python -m scripts.train
```

2. Quantitatively evaluates multiple DQN checkpoints (optional, for checkpoint analysis).

```
python -m scripts.evaluate_checkpoints
```

3. Detects false convergence in experiment results (optional, for quality control).

```
python -m scripts.detect_false_convergence
```

4. Loads experiment results, applies penalties to runs, and generates a penalized ranking JSON.

```
python -m scripts.penalize_runs
```

5. Selects the best model checkpoint based on penalized ranking and copies it to the best_model directory.

```
python -m scripts.select_best_model
```

6. Performs a quick test of the best model to ensure it loads and runs inference correctly.

```
python -m scripts.smoke_test_best_model
```

7. Runs a quantitative evaluation of the best model and outputs evaluation metrics.

```
python -m scripts.evaluate_best_model
```

8. Runs inference with the best model and collects agent trajectories for later analysis.

```
python -m scripts.collect_trajectories
```

9. Evaluates the best model under action noise (perturbed evaluation).

```
python -m scripts.perturbed_evaluation
```

10. Evaluates the best model in a stochastic environment (stochastic evaluation).

```
python -m scripts.stochastic_evaluation
```

## Visualization and Analysis

11. Visualizes agent trajectories from inference runs, using the best model and configuration.

```
python -m scripts.visualize_trajectories
```

12. Visualizes and analyzes experiment results and metrics.

```
python -m scripts.visualize_results
```

13. Compares experiment results, aggregating and analyzing metrics across runs.

```
python -m scripts.compare_results
```

## Rendering and Debugging

12. Renders the maze environment as ASCII art for visualization and debugging.

```
python -m scripts.render_maze_ascii
```

13. Renders the maze with a specific agent trajectory overlaid in ASCII art.

```
python -m scripts.render_maze_trajectory
```

## Utilities and Packaging

14. Runs inference with a trained model (can be used for custom inference runs).

```
python -m scripts.run_inference
```

15. Packages the final experiment results and artifacts for sharing or archiving.

```
python -m scripts.package_experiment
```
