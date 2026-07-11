# Packet-level validation protocol

The large statistical study uses the physically motivated proxy because a full
8-scenario × 30-repetition packet-level campaign is computationally prohibitive.
The packet-level experiment therefore checks calibration and claim transfer on
three representative scenarios:

| Scenario | Use case | Role |
|---|---|---|
| Dense urban eMBB | eMBB | Throughput-saturated boundary case |
| Highway URLLC | URLLC | Latency- and reliability-critical regime |
| Rural mMTC | mMTC | Fairness-critical regime |

## Calibration

For each scenario, evaluate fifteen configurations in both environments:

1. one expert heuristic;
2. one stored Llama prior;
3. one stored Qwen prior;
4. twelve random configurations.

Estimate an ns-3 throughput normalizer from these runs and calculate Spearman
rank correlation between proxy and packet-level scores. The reported values are
0.65, 0.59, and 0.70 for scenarios 1, 2, and 4, respectively.

## Matched-budget comparison

For each scenario, run:

- cold PSO;
- cold Gaussian-process Bayesian optimization;
- prior-guided PSO seeded by Qwen, Llama, and the heuristic;
- the heuristic as a one-call reference.

The manuscript uses a 60-evaluation budget and three optimizer repetitions.
The packet-level objective uses the same throughput, packet-delivery, delay,
fairness, and cross-phase robustness equations as the proxy objective.

## Interpretation

The packet-level experiment is a partial validation, not proof that the proxy
and 5G-LENA are equivalent. It preserves the seeded advantage in highway URLLC
and rural mMTC, while dense urban eMBB remains a boundary case where cold search
wins at the final budget.
