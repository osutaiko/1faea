# 🫪 Continuous emoji mixtures 🔬

Foreground diagnostic. No worker, training, model replacement, or English generation.

Using the existing encoder and frozen meaning vectors, compare hard states with continuously weighted emoji mixtures. Retrieve each answer's cached MiniLM representation among 100 candidates per split. All settings are reported; none is selected using test scores.

| Representation | Validation top-1 | Test top-1 |
| --- | --- | --- |
| Hard emoji choices | 72/100 | 80/100 |
| Full mixture, temperature 1 | 73/100 | 71/100 |
| Sparse top-2 mixture | 79/100 | 81/100 |
| Sparse top-4 mixture | 80/100 | 85/100 |
| Sparse top-8 mixture | 82/100 | 83/100 |
| Unrelated hard states | 0/100 | 1/100 |

Sparse mixtures show a modest descriptive improvement in this check. Broad mixtures dilute meaning. This supports testing sparse continuous emoji states, but does not establish statistical significance or conversational ability.

The encoder already sees the English answer. The reference uses the same semantic embedding space that initializes emoji meanings. Mean pooling cannot assess order, negation, relations, factual accuracy, or reasoning. Continuous weights may encode information that their displayed emoji summaries cannot explain. These are representation diagnostics, not chat accuracy.

Next: a bounded comparison using sparse mixtures in an ordered decoder. Keep actual question-answer evaluation separate from answer-side retrieval. Preserve the active chat model.
