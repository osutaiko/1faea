# ⚡ Positional-memory experiment 🫪

Two isolated clones start from the active baseline. The experiment adds learned memory-position embeddings to one clone. Both receive the same eight two-ID pairs in both orders, derived from existing cached emoji states. This is an identity-copy diagnostic, not generated English Q&A or a chat dataset. 📚🔬

After 200 updates per clone, the unordered decoder copies 8/16 training sequences exactly; the positioned decoder copies 16/16. On eight new pairs in both orders, results are 0/16 and 1/16 respectively. The final matched comparison takes about 14 seconds of compute. The preceding training-only check took about 18 seconds. ⏱️

Positions remove the architecture's permutation invariance and allow memorizing both orders. They do not establish a general copying algorithm, semantic roles or factual answering. Poor held-out copying remains a blocker. No model is promoted; the active checkpoint hash is unchanged. 🔒🚧

Next short diagnostic: compare an explicit pointer/copy output with the generative head under the same positional-memory and unseen-pair test. This separates retrieving supplied IDs from choosing their order. Repairing meaning-bearing answer targets remains necessary before general-answer training. 🧪

[Raw measurements](position-check.json)
