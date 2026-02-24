# AI Phantom

Proyecto RL sin gymnasium para resolver laberintos 2D.

## Quick start

1. Runs a quick test of the maze environment and BFS planner to ensure everything works.

```
python -m scripts.smoke_test
```

2. Trains a PPO agent in the maze environment (phase 0, no walls).

```
python -m scripts.train_ppo
```

3. Pretrains a behavioral cloning (BC) agent for phase 1 (with walls), using teacher trajectories.

```
python -m scripts.pretrain_bc_phase1
```

4. Trains a PPO agent in the maze environment (phase 1, with walls), possibly using BC warm-start.

```
python -m scripts.train_phase1
```
