# Population Coding of Orientation: Poisson-GLM Encoding and Decoding

Simulated V1-like population of N orientation-tuned neurons (von Mises tuning on the doubled angle, Poisson counts in a
200 ms window). Per-neuron **Poisson GLMs** (log link on cos 2θ, sin 2θ) are fitted on 3000 training trials; held-out
orientation (1000 trials) is decoded by **maximum likelihood** over the fitted GLMs and by a **population vector**.
Decoder error is compared with the **Cramér–Rao bound** from the Fisher information of the true independent-Poisson
population. A shared multiplicative gain (sd 0.3) adds noise correlations.

```
pip install numpy scikit-learn matplotlib
python popcoding.py     # writes results.json and decoding_error.png
```

## Results (RMSE in degrees, `results.json`)
| N | Cramér–Rao | GLM-ML (independent) | GLM-ML (shared gain 0.3) |
|---|---|---|---|
| 8 | 6.9 | 7.5 | 7.9 |
| 16 | 4.9 | 5.0 | 5.4 |
| 32 | 3.5 | 3.6 | 3.8 |
| 64 | 2.4 | 2.5 | 2.7 |

Error falls as ≈1/√N and the fitted-GLM decoder sits within ~10 % of the bound. Shared gain fluctuations cost roughly
5–10 % extra error. The population vector performs identically to ML here, because tuning is dense and uniform
(they differ with heterogeneous tuning, not tested).

## Caveats
Simulated data with a known tuning family (the GLM is well-specified), one random seed, no real recordings.
The Cramér–Rao bound ignores the gain noise, so it is not a bound for the correlated condition.
