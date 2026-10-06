# ⚡ Baseline quick checks 🫪

The active 5/12 checkpoint is unchanged, verified by SHA-256. All checks use existing cached states and targets; no new Q&A is generated. 📚🔒

- Replacing states with unrelated inputs changes 30/30 replies; zero states also change 30/30. The decoder uses its input, but this does not prove correct understanding.
- Reversing the eight states changes 0/30 replies. First-token logits differ by only 0.0000038. `GroundedReply` has no memory-position embeddings, so its cross-attention treats memory as an unordered collection. Input order cannot be recovered by this decoder.
- A disposable clone memorizes all 16 existing targets after 150 updates, versus 1/16 before training. Teacher-forced loss falls from 3.225 to 0.0199. Updates take about 9 seconds. This demonstrates narrow optimization, not correct meanings or generalization.
- 21/30 normal replies contain one symbol. Forcing at least three symbols mostly adds repeated or unrelated emojis: 🦸 becomes 🦸🦸🦸; 🥩-style topics remain topic fragments. This is not an answer-quality solution.
- The previously reviewed targets still contain errors: 117→💯, yards→🥖, musical strings→🧵. Memorizing them would teach incorrect literal meanings.

Next short experiment: add state-position information to an isolated decoder clone and compare ordered-memory sensitivity and tiny-set learning. Repair answer representations before scaling. Keep the current model active; no diagnostic model is promoted. 🧪

[Full measurements](quick-checks.json)
