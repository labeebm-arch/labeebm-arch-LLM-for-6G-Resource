# Result-file roles

- `model_campaigns/`: 30-run Llama and Qwen campaigns used for the standalone
  one-shot/few-shot and sample-efficiency results in Table 3.
- `seed_bank/`: exact 10-run historical Llama and Qwen files used as the cyclic
  prior source in the seeded study. See `PROVENANCE.json`.
- `seeded/`: 30-repetition cold and prior-guided optimizer histories used for
  Tables 4--6 and Figure 5.
- `ns3/`: immutable packet-level reference summaries.
- `prior_ns3/`: historical optimizer curves used for Figure 1.
- `reproduced/`: default destination for newly generated outputs; gitignored.

The 30-run model campaign and 10-run seed bank are intentionally distinct and
must not be substituted for one another.
