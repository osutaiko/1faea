# 🔍 Answer-target audit 🫪

The best five-hour checkpoint remains unchanged. This audit recomputes its answer targets from the existing 9,016 training, 100 validation and 100 test examples. Checkpoint and feature-file hashes are recorded in the JSON report. No new labels or Q&A were generated. 📚🔒

Training targets retain 5,085 of 5,404 exact core-name matches (94.10%). This is string-match retention, not meaning accuracy: “key” in “key features” can still incorrectly assert 🔑. 14,770 anchor IDs are derived without an exact core match. 5,707/9,016 answers have no exact core match. These counts do not establish that unmatched answers are unrepresentable. 🔢⚠️

1,763/9,016 targets contain one symbol; 2,073 reach the six-symbol cap. There are 7,693 distinct target sequences and 432 sequences shared by distinct English answers. Shared targets can represent paraphrases, so collision counts alone do not prove errors. 📊

Inspection of 20 seeded validation examples finds concrete content loss and misleading meanings. “117” becomes 💯; “100 yards” becomes 💯🥖; musical strings become 🧵; the chess winner becomes ♟️ without a name. Canada’s two national sports are one useful coarse answer: 2️⃣🇨🇦🥍🏒. Topic similarity does not guarantee an answer. This qualitative review is not a general accuracy score or human-validated dataset. 🧪
xzcz
The target function prepends heuristic associations, removes repeated IDs globally, and caps content at six symbols. This discards multiplicity and can destroy numbers, lists and relationships. More training on these labels is not justified. 🚧

Next gate: test answer-side reconstruction on the premade data before another answering run. Preserve ordered, repeated emoji states and test a training-only English reconstruction decoder with no English answer prefix. Compare matched, shuffled and empty states on held-out data. Successful reconstruction would establish information retention only; factual answering and readable emoji composition still need separate evaluation. Runtime must continue to generate direct emojis, with no English answer decoder. 🫪🏋️

[Full reproducible audit](target-quality-audit.json)
