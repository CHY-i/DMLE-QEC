# Detector error models

Each code setting contains the four DEMs used to generate neural-decoder
training data:

- `si1000.dem`: SI1000 baseline.
- `ca.dem`: correlation-analysis estimate.
- `rl.dem`: decoder-specific RL-optimized estimate.
- `aligned.dem`: likelihood-aligned DEM studied in this work.

The directory name records the code family, distance, and number of rounds.
All reported neural-decoder comparisons use the corresponding four files under
identical training settings.
