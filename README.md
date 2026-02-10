# AI Phantom

Supposedly execution order.

---

1. Trains a DQN agent in the maze environment using configuration files and saves checkpoints.

```
python -m scripts.train
```

2. Loads experiment results, applies penalties to runs, and generates a penalized ranking JSON.

```
python -m scripts.penalize_runs
```

3. Selects the best model checkpoint based on penalized ranking and copies it to the best_model directory.

```
python -m scripts.select_best_model
```

4. Performs a quick test of the best model to ensure it loads and runs inference correctly.

```
python -m scripts.smoke_test_best_model
```

5. Runs a quantitative evaluation of the best model and outputs evaluation metrics.

```
python -m scripts.evaluate_best_model
```

6. Runs inference with the best model and collects agent trajectories for later analysis.

```
python -m scripts.collect_trajectories
```

7. Visualizes agent trajectories from inference runs, using the best model and configuration.

```
python -m scripts.visualize_trajectories
```

8. Renders the maze environment as ASCII art for visualization and debugging.

```
python -m scripts.render_maze_ascii
```

9. Renders the maze with a specific agent trajectory overlaid in ASCII art.

```
python -m scripts.render_maze_trajectory
```

10. Compares experiment results, aggregating and analyzing metrics across runs.

```
python -m scripts.compare_results
```
