# 🫪

🫪 **[osutaiko/1faea](https://github.com/osutaiko/1faea)** 🫪 | **U+1FAEA: distorted face** 🫪

🧠 💬 🚀 🔬 🧪 🤖 🌈 ✨ 🎉 📚 🧩 💡 🎯 📊 🔥 🛠️ ⚙️ 💻 🫪 🛠️ ⚙️ 💻 🎯 📊 🔥 📚 🧩 💡 🌈 ✨ 🎉 🔬 🧪 🤖 🧠 💬 🚀

### 🤖 Local Discord bot 🫪

`emoji_discord_bot.py` responds only to messages in channels named `🫪` in any
server it joins. Each message is handled independently; it keeps no conversation
history and uses a local Qwen2.5-1.5B-Instruct model. The model
is downloaded on first run into the ignored `.hf-cache/` folder; model weights
are not included in this repository. A tokenizer grammar hard-limits generation
to catalog emoji tokens. A few fixed prompt examples guide answer composition.
This is an experimental emoji-output bot, not an emoji-native reasoning model.

Create a Discord application, enable **Message Content Intent**, and invite the
bot with View Channels and Send Messages permissions. Set its token in PowerShell
and start it with:

```powershell
$env:DISCORD_BOT_TOKEN = "your-discord-bot-token"
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -X utf8 emoji_discord_bot.py
```

## Current development checkpoint 🫪 🧠 💬 🧪

🧩📚 The new **meaning-map + premade-data** path uses a one-time map of 3,951
emojis and existing UltraChat English answers. It generates **zero new Q&A labels**.
The map retains literal core meanings and separate associations; generation and
automated auditing cost approximately **$0.56**. See [meanings](data/emoji-meanings/README.md). 😋🌈🔍

🏋️🧠 The first bounded experiment uses 102 training, three validation, and 11 test
pairs. Frozen semantic description embeddings initialize the emoji vocabulary.
An English reconstruction decoder is used only during training. Runtime reads
English questions, commits hard emoji IDs, and generates emojis directly.
No English assistant answer is generated at runtime. 📚➡️🧩➡️💬

🧪📊 Raw word-vector trials collapsed. Semantic initialization avoids that collapse:
11 test questions produce 11 distinct replies using 22 emojis. However, it achieves
**0/11 exact learned reference-state matches**, zero literal answer-anchor recall,
and worse reconstruction than zeroed states. These are diagnostic metrics, not
general-chat accuracy. Replies remain unreliable; this prototype is **not promoted**.
The earlier chat checkpoint remains active. All **73 tests pass**. 🚧🔬🫪

🔍 [Method](data/emoji-grounded/README.md) · [Results](data/emoji-grounded/result.json)
· [Dataset](data/emoji-grounded/report.json) · [Map audit](data/emoji-meanings/report.json) 📚💾✨

```powershell
# One-time map asset; an existing map is not regenerated.
.venv\Scripts\python.exe -X utf8 emoji_meanings.py
.venv\Scripts\python.exe -X utf8 emoji_grounded_data.py --pages 20
.venv\Scripts\python.exe -X utf8 emoji_grounded_semantics.py
.venv\Scripts\python.exe -X utf8 emoji_grounded_fit.py --name semantic-v1 --grounding-steps 300 --reconstruction-steps 20 --reply-steps 800
# Inspect the experimental checkpoint. Each question is independent.
.venv\Scripts\python.exe -X utf8 emoji_grounded_chat.py --question "How can I stay dry in the rain?"
```

📚 🌱 The standalone dataset pipeline prepares 200 public user questions:
160 training, 20 validation, and 20 test prompts. It defines no question categories.
[Teacher instructions](data/standalone/TEACHER.md) request direct compositional emoji
states and answers. English readings are generated afterward for auditing.
This assistant generated 200 direct emoji candidates. A separate audit by the same
assistant rejected 48 ambiguous answers and quarantined seven remaining training
examples that overlap held-out requests. The provisional dataset has 111 training,
16 validation, and 18 test rows. This is not independent verification. 🧪🔍🫪

🏋️ 📊 A 2,000-step standalone trial reaches **0/10 exact matches on answerable test questions**
and matches 4/8 abstention targets. Its 22.2% overall
score is below the 44.4% always-unknown baseline. It is not promoted.
More varied, verified supervision is needed; more steps on this tiny set did not help.
Exact matching can miss valid alternative answers, and these targets remain provisional.
All 73 implementation tests pass. 🚧❓🧪✅

🤖 📚 The API runner generates direct emoji candidates and reviews them with a
distinct model in separate requests, using [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
The first external pilot uses GPT-4.1 mini generation and GPT-4.1 review on 20 questions.
The reviewer approves 11, including two unknown answers. Successful requests cost an
estimated $0.063 at published token prices. Review still accepts invented meanings:
brain as genetic transmission and black circle as switching off a phone.
These labels are not promoted or used for training. See [pilot results](data/standalone/api/pilot-report.json). 🧪💸🔍🚧
Automated review remains a model judgment, not verified ground truth. 🔍🫪

🔬 A hosted direct-answer prototype keeps the full question in one `gpt-6-luna`
call and asks for an emoji-only final answer; it does not pass through an emoji
summary. A small eight-prompt smoke test produced plausible short replies for
facts, negation, relations, advice, and emotional support. One non-emoji symbol
was rejected, then succeeded on retry. This is not a benchmark or a standalone
emoji-native model: the hosted LLM still uses its ordinary internal representations.
It has no conversation history, and API storage is disabled. [Outputs and limits](data/emoji-grounded/direct-answer-quick-check-2026-10-07.json).
Try it with `OPENAI_API_KEY` set:

```powershell
.venv\Scripts\python.exe -X utf8 emoji_api_chat.py --direct
```

📚🔍 The expanded run covers 100 questions. One invalid candidate is quarantined.
GPT-5.4 mini reviews fixed catalog meanings, without an English answer generated
before the emojis. It passes seven calibration checks and accepts 33/99 candidates.
Three training prompts overlap held-out requests. The provisional set contains
22 training, four validation, and four test examples. 🧩🛡️📊

🏋️🧪 An 800-step trial matches **0/3 answerable test targets** and the one unknown
target. It equals the 25% always-unknown baseline and is not promoted.
The sample is tiny, and calibration does not establish reliable review on every question.
All successful API requests, including exploratory reviews, cost an estimated **$0.372**.
See [review](data/standalone/api/review-literal-gpt-5.4-mini/blind-review-report.json),
[dataset](data/standalone/api/dataset-report.json), [training](data/standalone/api/training-result.json),
and [cost](data/standalone/api/cost-report.json) reports. 💸📚🔬🫪

```powershell
.venv\Scripts\python.exe -X utf8 conversation_standalone_data.py prepare
.venv\Scripts\python.exe -X utf8 conversation_standalone_data.py validate teacher-results.jsonl
.venv\Scripts\python.exe -X utf8 conversation_standalone_bootstrap.py
.venv\Scripts\python.exe -X utf8 conversation_standalone_fit.py
# Requires a valid OPENAI_API_KEY and two accessible structured-output models.
.venv\Scripts\python.exe -X utf8 conversation_standalone_teacher.py --generator GENERATOR_MODEL_ID --reviewer REVIEWER_MODEL_ID --limit 20
.venv\Scripts\python.exe -X utf8 conversation_standalone_review.py
.venv\Scripts\python.exe -X utf8 conversation_standalone_compile.py
.venv\Scripts\python.exe -X utf8 conversation_standalone_fit.py --dataset data/standalone/api/provisional.jsonl --name api-reviewed --steps 800
# Use the earlier checkpoint without carrying history between questions.
.venv\Scripts\python.exe -X utf8 conversation_web.py --independent
```

🔍 ✅ Validation checks format and emoji vocabulary. Semantic review and cross-split
similarity checks remain necessary for future production training. API candidates
are separate from the provisional bootstrap labels. 📚🧠

🧠 🔤 🌈 The new curriculum adds 16 everyday intents, quantities from zero to nine,
mixed attributes, and user corrections. It has 1,408 training 🏋️ rows, 224 validation ✅
rows, and 416 test 🔍 rows. Supplemental entity pools are frozen and disjoint.
All 3,953 emoji sequences remain available. Availability does not establish mastery. 🫪✨

📚 🧩 💡 Training also uses 38 reviewed teacher user paraphrases, 256 number-wording
variants, and 23 reviewed public user inputs. Targets are authored directly as emojis.
Public assistant answers are not training targets. 🏋️🤖📚

⚡ 🛠️ 🧠 A small causal emoji processor learns from frozen pretrained emoji features.
It rebuilds features from integer IDs at every reply step. A validation-selected input
blend preserves earlier skills. The input backbone still reads ordinary text.
Each stage uses continuous numerical activations; only committed emoji states persist. 🔢🫪💬

| Frozen checkpoint evaluation 🔍 | Rows/turns | Exact replies 🎯 |
| --- | ---: | ---: |
| New combinations, supplied history 🧩 | 416 | 62.0% |
| Ordinary numbers, supplied history 🔢 | 416 | 61.3% |
| Ordinary numbers, own-history stress sessions 💬 | 384 | 6.8% |
| Earlier curriculum, supplied history 📚 | 2,189 | 85.4% |
| Public supported requests 📚 | 5 | 60.0% |
| Public unsupported requests: correct abstention ❓ | 27 | 14.8% |

🧪 🔍 These are restricted-language scores. Public labels check coarse intent and scope,
not whether an answer fully solves the request. Selection used validation only.
The final test prompts did not select or train this checkpoint. ✅📊🫪
Earlier-curriculum correct state-plus-reply traces improve from 73.3% to 81.1%.
All evaluated outputs terminate. 🎯✅🔚
The earlier checkpoint scores 7.7% on the same number-wording test and 4.7% on
the same own-history stress sessions. Both remain poor at complete conversations. 💬🚧🫪
The 32 stress sessions run all three contrast queries in sequence. Training uses
the same snapshot for these queries and only one query in each continuation. 💬🔄🧩

⚡ ⏱️ The median emoji reply stage takes 0.021 seconds, versus 1.04 seconds for its
pretrained-feature parent, on 12 identical validation memories. Both answer 11 correctly.
This benchmark excludes text encoding and model loading. Full local smoke turns take
roughly 0.2–0.3 seconds. 🏃💬✨

🚧 🧠 The model is **not ready for general use**. Generated history quickly degrades
replies. Unsupported requests often receive guesses. Next experiments should train on
generated emoji transcripts, strengthen user-fact handling, and calibrate the unknown
state. The next checkpoint needs a fresh untouched test set. 🧪📚❓🔄

```powershell
# Open the local, single-session playground 🫪💬
.venv\Scripts\python.exe -X utf8 conversation_web.py
# Use the same checkpoint in the terminal 🧠🔤
.venv\Scripts\python.exe -X utf8 conversation.py --experiment conversation-compositional --semantic-encoder chat --name reliability-selected
# Create data, then train the supplemental and reviewed adaptations 📚🏋️
.venv\Scripts\python.exe -X utf8 conversation_reliability_data.py
.venv\Scripts\python.exe -X utf8 conversation_topic_curate.py
.venv\Scripts\python.exe -X utf8 conversation_public_scope.py
.venv\Scripts\python.exe -X utf8 conversation_reliability_fit.py
.venv\Scripts\python.exe -X utf8 conversation_reliability_fit.py --name reliability-reviewed --parent reliability --steps 3200 --seed 439 --learning-rate 0.00005 --reviewed
.venv\Scripts\python.exe -X utf8 conversation_emoji_features.py
.venv\Scripts\python.exe -X utf8 conversation_reliability_select.py --blend-only --blend-name reliability-blended75 --reviewed-weight 0.75
# Evaluate validation before selecting; include every candidate with completed reports ✅
.venv\Scripts\python.exe -X utf8 conversation_reliability_evaluate.py --name reliability-blended75 --split validation --live
.venv\Scripts\python.exe -X utf8 conversation_reliability_select.py --names selected reliability-blended75
# Audit runtime and memory; neither command trains or selects models 🔍🧠
.venv\Scripts\python.exe -X utf8 conversation_runtime_benchmark.py
.venv\Scripts\python.exe -X utf8 conversation_memory_audit.py
```

📦 💾 Local checkpoints, reports, and the model card live in
`runs/conversation-compositional/reliability-selected/`. Checkpoints are excluded from Git.
The scripts require the earlier pretrained cache and experiment artifacts.
The archived checkpoint records this run; rerunning the workflow can produce different scores.
All 60 implementation tests pass. Tests do not establish language quality. 🧪✅🫪

✨ 🫪 🌈 🫪 ✨

## Earlier conversation prototype 🧠 💬 🚀 🫪

🔬 🧪 🤖 `conversation.py` reads text, replies in emojis, and keeps
emoji-only conversation 💬 memory 🧠. It uses frozen
[SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
with separately trained input and reply decoders. It generates no English answer to translate. The input decoder 🔓 selects
atomic emoji states; the reply decoder 🔓 receives only those states and emoji
history. Each output step rebuilds features from hard IDs. It retains no hidden states or KV cache.

🌈 ✨ 🎉 The vocabulary 🔤 contains all 3,953 Unicode Emoji 17.0 sequences/components.
The conversation 💬 curriculum covers 42 social, emotional, planning, and simple
factual intents, plus preferences, corrections, selective recall, colors,
locations, pronouns, choices, and small addition problems. See the state grammar in [data/conversation/GRAMMAR.md](data/conversation/GRAMMAR.md).
Neural activations remain continuous.
Committed states and memory 🧠 are discrete. This prototype 🧪 does not prove symbolic computation or general reasoning 🧩.

✨ 🫪 🌈 🫪 ✨

📚 🧩 💡 The second curriculum has 10,634 training 🏋️, 1,116 validation ✅, and 2,189 test
rows. Emoji targets are authored directly. An offline
[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
experiment generated English **user-message paraphrases**; reviewed candidates
and additional authored input templates added 788 inputs. None exactly match normalized validation ✅ or test prompts. We rejected teacher answers,
perspective changes, lost placeholders, and altered facts.
Approved indices are bound to the reviewed source file's SHA-256. Conversation 💬 inference does not load this teacher. The earlier experiment used the downloaded
[Everyday Conversations](https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k)
corpus for an unscored natural-prompt audit 🔍.
Model revisions and source metadata are in `data/conversation/`.

🎯 📊 🔥 Validation ✅ loss selects each run's input checkpoint 💾. Greedy validation ✅ selects the output head:

| Input encoder | Exact emoji state on 1,116 validation rows |
| --- | ---: |
| Compositional baseline | 56.9% |
| Reviewed input augmentation | 81.0% |
| Augmentation with named-embedding output scores | 82.3% |

🛠️ ⚙️ 💻 These scores use cached FP8 pretrained features. Runtime evaluation uses fresh
features. The selected input checkpoint 💾 is step 1,800 of its 3,000-step
adaptation. Its output scores use frozen named emoji embeddings rather than
independent vocabulary 🔤 classifier rows. A separate 3,000-step memory 🧠 adaptation
adds 768 examples with varied fact order, either queried object, and intervening
social turns; 128 examples validate it. The selected
memory 🧠 checkpoint 💾 is step 800. `runs/conversation-compositional/selected/`
holds paired checkpoints 💾 and evaluation artifacts.

```powershell
# Interactive session: /reset clears memory; /quit exits.
.venv\Scripts\python.exe -X utf8 conversation.py --experiment conversation-compositional --semantic-encoder chat --name selected
# Show the committed states as well as the emoji replies:
.venv\Scripts\python.exe -X utf8 conversation.py --experiment conversation-compositional --semantic-encoder generate --name selected --trace "I like pizza." "What do I like?"
# Reproduce curriculum features, initial training, then the reviewed adaptations:
.venv\Scripts\python.exe -X utf8 conversation.py --experiment conversation-compositional prepare --curriculum compositional
.venv\Scripts\python.exe -X utf8 conversation.py --experiment conversation-compositional train --steps 6000
.venv\Scripts\python.exe -X utf8 conversation_encoder_fit.py --name baseline --score-only
.venv\Scripts\python.exe -X utf8 conversation_curate.py
.venv\Scripts\python.exe -X utf8 conversation_encoder_fit.py --name augmented
.venv\Scripts\python.exe -X utf8 conversation_encoder_fit.py --name semantic --semantic
.venv\Scripts\python.exe -X utf8 conversation_memory_fit.py --name augmented
.venv\Scripts\python.exe -X utf8 conversation_select.py
# Experimental longer-dialogue adaptation; keep the earlier checkpoint:
.venv\Scripts\python.exe -X utf8 conversation_dialogue_fit.py
.venv\Scripts\python.exe -X utf8 conversation_semantic_audit.py
.venv\Scripts\python.exe -X utf8 -m unittest test_model test_reconstruction test_semantic test_semantic_finetune test_semantic_pointer test_emoji_lm test_emoji_full test_conversation
```

🧠 💬 🚀 The selected short-memory 🧠 checkpoint 💾 achieved 79.8% exact replies, 76.8% exact
input states, and 73.3% correct state-plus-reply traces on all 2,189 test rows.
Replies given the intended state reached 92.5%; all replies terminated.
The baseline scored 53.3%, 40.8%, and 39.5%, respectively. These scores assume correct history. They do not measure general conversation 💬 accuracy 🎯.

✨ 🫪 🌈 🫪 ✨

🔬 🧪 🤖 The additional `dialogue` checkpoint 💾 trains on 960 longer histories with 240
validation ✅ examples, including incorrect assistant replies after user
corrections. It reached 80.4% exact replies and 74.1% correct state-plus-reply
traces on the same 2,189-row test. Its 17-turn development smoke test improved from 10 to 13 correct
replies, but the eight swapped-color checks regressed from eight to three.
Neither variant follows color/location query operators reliably when both
attributes appear in the same history: changing only the operator changed no
replies in four tested pairs. Correct replies can therefore mask incorrectly
interpreted states. The 51 tests check implementation and data invariants. They do not prove language quality or reliable reasoning 🧩.

🌈 ✨ 🎉 Text input is limited to 128 pretrained tokens, memory 🧠 to 128 emoji IDs, and
each generated state/reply to 32 symbols. Old whole turns are evicted before
the next input exceeds the memory 🧠 window. Literal emojis and exact Unicode
names ground entity copying; ordinary aliases and plurals can be missed.
The row benchmark supplies correct prior emoji history. `conversation_audit.py`
separately tests history produced by the model itself, swapped color facts,
and public natural prompts. Full test accuracy 🎯, oracle-state accuracy 🎯, and
correct-state-plus-reply accuracy 🎯 must be distinguished. Shared templates,
small controlled arithmetic, and synthetic facts about arbitrary emoji entities
limit what this benchmark establishes. The original social-only run achieved
70% exact replies but only 54.4% correct states on 810 test rows, despite 100%
oracle-state replies. Open-ended conversation 💬 remains unreliable. This model is not ready for general use.

## Full Unicode vocabulary 📚 🧩 💡 🫪

🎯 📊 🔥 `emoji_full.py` is the spatial reasoning 🧩 experiment. Its alphabet contains all
3,953 fully qualified emoji sequences and components in the official
[Unicode Emoji 17.0 catalog](https://www.unicode.org/Public/17.0.0/emoji/emoji-test.txt),
including flags, skin tones, genders, and joined sequences. Each sequence has one atomic integer ID. Alternate qualification spellings are not
separate tokens; vendor artwork is not a separate vocabulary 🔤. Names and operator meanings are in
`data/emoji-full/catalog.json`. Emoji names initialize frozen symbol embeddings;
runtime reasoning 🧩 generates no intermediate English answer.

🛠️ ⚙️ 💻 Nine IDs serve as operators or controls. The other 3,944 IDs can name entities
in facts and questions. Training 🏋️ uses 7,888 generated two-fact examples covering
every entity emoji. Validation ✅ and test use 256 and 512 new combinations.
The decoder 🔓 also receives learned 🌱 pairwise ID-equality features. These preserve identity when name embeddings are similar. Equality is an input feature; there is no hand-written
relation solver. Generation predicts one symbol per step. Each step rebuilds features from hard IDs.

✨ 🫪 🌈 🫪 ✨

🧠 💬 🚀 The initial expanded-vocabulary 🔤 run achieved 47.1% exact test continuations.
After adding identity features, the checkpoint 💾 selected at step 1,000 of a
2,000-step run achieved 100% exact, correctly terminated continuations on both
emoji-state suites. These results 📊 cover held-out combinations. They do not cover unseen IDs or general language understanding.

🔬 🧪 🤖 The clause parser was separately adapted on 2,048 clauses mixing official
Unicode names and literal emoji glyphs across 40 left/right sentence forms.
It was selected at step 600 of 1,000 using 256 validation ✅ clauses in four
reserved forms. End-to-end output accuracy 🎯 was 100% on 93 name prompts and 96
glyph prompts. Exact parsed-state accuracy 🎯 was 92/93 and 96/96, respectively;
One incorrect name parse left the output unchanged. Names with
sentence punctuation are excluded from that check; use their emoji glyphs.

```powershell
.venv\Scripts\python.exe -X utf8 emoji_full.py generate "The 🚲 is left of the 🧑🏽‍🚀. The 🧑🏽‍🚀 is left of the 🇰🇷. Is the 🚲 left of the 🇰🇷?"
# Reproduce training using the existing pretrained/parser checkpoint:
.venv\Scripts\python.exe -X utf8 emoji_full.py prepare
.venv\Scripts\python.exe -X utf8 emoji_full.py train --steps 2000
.venv\Scripts\python.exe -X utf8 emoji_full.py adapt-parser --steps 1000
.venv\Scripts\python.exe -X utf8 -m unittest test_model test_reconstruction test_semantic test_semantic_finetune test_semantic_pointer test_emoji_lm test_emoji_full
```

🌈 ✨ 🎉 Checkpoints 💾 and metrics are in `runs/emoji-full/`. All 41 tests pass, including
full-vocabulary 🔤 copying, atomic joined sequences, training 🏋️ coverage, held-out
combination separation, and renaming-invariant ID-equality features.
This remains a conditional language model for two spatial facts and one query.
Its larger alphabet does not establish reasoning 🧩 about emotions, travel, food,
or other topics represented by those emojis, and it is not ready for general use.

## Autoregressive emoji model 📚 🧩 💡 🫪

🎯 📊 🔥 Run the generative experiment with `emoji_lm.py`. It parses a text question and discards its text features. It then predicts an emoji programme, one symbol per step. It outputs a four-symbol inferred fact, then a verdict. For example:

✨ 🫪 🌈 🫪 ✨

```text
📌🐱⬅️🐦✅
```

🛠️ ⚙️ 💻 The first four symbols assert that the cat is left of the bird. The final
symbol answers the input query. The 13 embedded symbols have fixed meanings;
`▶️` begins generation and `🔚` terminates it. The start symbol is input-only. Normal output hides the end symbol. The model learns the
positions of fact markers, arguments, relation, verdict, and ending symbol.

🧠 💬 🚀 The next-symbol head uses two causal transformer decoder 🔓 layers, initialized
from the successful semantic pointer decoder 🔓. It combines learned 🌱 vocabulary 🔤 scores with attention that copies source entities.
Next-symbol loss trains its verdict. Generation never calls the pointer classifier's deduction or answer methods. SmolLM2-135M supplies frozen
features computed from the selected emoji source state. Each next-symbol step
rebuilds those features and the prefix from integer IDs, without retaining
hidden states or a key/value cache. The only ordinary-text computation is the
independent clause parsing before the initial emoji state is selected.

🔬 🧪 🤖 Training 🏋️ uses 384 grounded states and their six-symbol target continuations,
including the ending symbol. Validation ✅ uses 96 held-out states. The selected
checkpoint 💾 is step 2,000 of a 2,000-step CPU run, chosen by next-symbol validation ✅
loss. Greedy generation produces exact, correctly terminated programmes on
all 96 examples in each of the validation ✅, test, wording-challenge, and audit
suites. These reuse the parser suites below. They test controlled grammar, not general language quality.

```powershell
.venv\Scripts\python.exe -X utf8 emoji_lm.py generate "The cat is left of the dog. The dog is left of the bird. Is the cat left of the bird?"
.venv\Scripts\python.exe -X utf8 emoji_lm.py generate "The cat is left of the dog. The dog is left of the bird. Is the cat left of the bird?" --trace
# Training requires the semantic pointer checkpoint described below:
.venv\Scripts\python.exe -X utf8 emoji_lm.py prepare
.venv\Scripts\python.exe -X utf8 emoji_lm.py train --steps 2000
.venv\Scripts\python.exe -m unittest test_model test_reconstruction test_semantic test_semantic_finetune test_semantic_pointer test_emoji_lm
```

🌈 ✨ 🎉 Training 🏋️ records live in `data/emoji-lm/`; the checkpoint 💾 and detailed results 📊
live in `runs/emoji-lm/`. All 37 tests pass, including normalized next-symbol
probabilities, causal teacher forcing, fresh per-symbol computation from hard
IDs, and a guard against borrowing the classifier's answer. This is a working
conditional emoji language model for the four-entity, two-fact spatial domain;
it remains far smaller in scope than a general-purpose assistant.

✨ 🫪 🌈 🫪 ✨

## Current source-copying prototype 📚 🧩 💡 🫪

🎯 📊 🔥 Run the strongest current experiment with `semantic_pointer.py`. It reads
ordinary text with SmolLM2-135M and emits fixed-meaning emoji states directly.
Inference uses no intermediate text or relation solver.
The pretrained backbone is frozen; learned 🌱 attention heads select source
mentions and fact endpoints. Only integer emoji IDs cross reasoning 🧩 stages.

🛠️ ⚙️ 💻 The text stage reads each of the three clauses separately. A lexical matcher
grounds the named animals; a shared neural pointer head learns which of two
mentions is the left endpoint. The other mention is the right endpoint.
The head sees pretrained contextual features, normalized pretrained token
embeddings, and learned 🌱 positions relative to the mentions. The deduction head
selects endpoints from the four entities present in the two emoji facts. It
never sees the original text. The answer head is reused from the frozen
grounded baseline and sees only the expanded emoji state.

🧠 💬 🚀 Training 🏋️ uses 480 automatically labeled clauses across 40 sentence forms and
384 emoji-state examples from the original training 🏋️ chains. Parser validation ✅
uses 48 clauses across four reserved forms. The selected checkpoint 💾 is step
200 of a 1,200-step run, chosen using validation ✅ loss. Earlier text tests now serve as regression checks. The expanded corpus includes their syntax.

| Suite | Examples | Correct answers | Fully correct traces |
| --- | ---: | ---: | ---: |
| Earlier validation regression | 96 | 100% | 100% |
| Earlier test regression | 96 | 100% | 100% |
| Reserved wording challenge | 96 | 100% | 100% |
| Additional audit after checkpoint selection | 96 | 100% | 100% |

🔬 🧪 🤖 The challenge was reserved before clause training 🏋️ but inspected during later
architecture iterations. The additional wording audit was defined and run
after the final checkpoint 💾 was selected, with no further training 🏋️. Neither
suite's clauses appear in parser training 🏋️ or validation ✅. These test controlled grammar, not arbitrary English. Entity-renaming
consistency is 100% across 2,304 emoji states; isolated inferred-fact reversal
probes are correct for all 16 applicable pairs in each suite. All 37 project tests
pass, including discrete boundaries, positional gradients, padding invariance,
and checks that correct answers cannot conceal incorrect traces.

✨ 🫪 🌈 🫪 ✨

🌈 ✨ 🎉 `correct_trace` requires the parsed facts/query, canonical inferred fact, and
answer all to match the labels. `novel_proof` requires that the inferred fact
is entailed and is absent from the original facts. Neither metric proves the answer head uses the inferred fact when original facts remain present.

```powershell
.venv\Scripts\python.exe -X utf8 semantic_pointer.py generate "The cat is left of the dog. The dog is left of the bird. Is the cat left of the bird?" --trace
# To train again, first train the grounded baseline if its answer checkpoint is missing:
.venv\Scripts\python.exe -X utf8 semantic.py train
.venv\Scripts\python.exe -X utf8 semantic_pointer.py prepare
.venv\Scripts\python.exe -X utf8 semantic_pointer.py train --steps 1200
.venv\Scripts\python.exe -m unittest test_model test_reconstruction test_semantic test_semantic_finetune test_semantic_pointer
```

📚 🧩 💡 Actual selected states for a positive query:

```text
📌🐱⬅️🐶 📌🐶⬅️🐦 ⚪⚪⚪⚪ 🔍🐱⬅️🐦
📌🐱⬅️🐶 📌🐶⬅️🐦 📌🐱⬅️🐦 🔍🐱⬅️🐦
✅
```

🎯 📊 🔥 Normal generation prints only the answer emoji; `--trace` prints diagnostic
JSON. Prepared data are in `data/semantic-pointer/`; the current checkpoint 💾,
evaluation, and smoke examples are in `runs/semantic-pointer/`. Intermediate
experiment results 📊 are preserved in `runs/semantic-pointer-v1` through `v4`.

🛠️ ⚙️ 💻 The supported domain is two separate positive left/right fact sentences
forming a three-entity chain, followed by one left/right question comparing
distinct entities, using the
named cat, dog, bird, and fish. The fourth entity supports unknown queries.
The grammar and marker tokens are fixed in code. This is a typed neural reasoner
backed by a pretrained LLM, not yet a general autoregressive emoji language
model. Computation within stages stays continuous. Discrete states carry information between stages.

✨ 🫪 🌈 🫪 ✨

## Semantic reasoning prototype 🧠 💬 🚀 🫪

🔬 🧪 🤖 `semantic.py` implements the original grounded baseline: text -> selected semantic
emoji IDs -> a fresh pretrained pass over those IDs -> selected derived fact ->
a fresh pretrained pass -> emoji answer. It generates no text reasoning 🧩 trace or translated answer. Only integer symbol IDs cross stages;
later stages receive no original-text features or key/value cache.

🌈 ✨ 🎉 The vocabulary 🔤 has fixed meanings: `🐱🐶🐦🐟` identify cat, dog, bird, and fish;
`📌A⬅️B` asserts A is left of B; `🔍A⬅️B` asks that relation; `⚪⚪⚪⚪`
marks an unused fact slot. `✅`, `❌`, and `❓` mean proven true, proven false,
and unknown from the supplied facts. This typed grammar is fixed in code. The
neural parser selects entities, the neural transition selects the endpoints of
one inferred fact, and the neural answer head selects the verdict.

📚 🧩 💡 The frozen baseline reuses SmolLM2-135M weights with separate trained adapters.
Meaning-word embeddings initialize atomic emoji embeddings once. Emoji stages directly use those embedding rows; they do
not tokenize words or generate text. Each stage computes continuously. Only states are discrete; activations are not emojis.

🎯 📊 🔥 Training 🏋️ uses 2,304 automatically labeled examples across six sentence forms,
with supervised intermediate states. Validation ✅ and test each contain 96
examples with held-out entity chains and new wording. The deduction adapter
uses attention over individual emoji tokens with learned 🌱 positional embeddings.
The answer adapter also trains on states containing only the inferred fact and
query, including reversed facts with recomputed labels. Evaluation measures
whether reversing that isolated fact changes the verdict correctly.
A relation checker generates labels and evaluates proofs; generation
does not call it. Frozen features are cached during training 🏋️. Inference computes
each stage afresh from the preceding selected symbols.

✨ 🫪 🌈 🫪 ✨

```powershell
.venv\Scripts\python.exe -X utf8 semantic.py train
.venv\Scripts\python.exe -X utf8 semantic.py generate "The cat is left of the dog. The dog is left of the bird. Is the cat left of the bird?" --trace
.venv\Scripts\python.exe -m unittest test_model test_reconstruction test_semantic
```

🛠️ ⚙️ 💻 Generated data live in `data/semantic/`; the adapter checkpoint 💾 and measured
results 📊 live in `runs/semantic/`. Cache the pretrained model in `.hf-cache`. This experiment supports four entities, two chain facts, one
deduction, and one relation query. It is an architecture prototype 🧪, not yet a
general-purpose language model. The earlier experiments below use private
codes and remain available for comparison.

🧠 💬 🚀 The original 2,000-step CPU run achieves 100% training 🏋️ accuracy 🎯, but generalization
fails: complete parsing accuracy 🎯 is 0% on both held-out wording sets. Final
answer accuracy 🎯 is 47.9% on validation ✅ and 64.6% on test (always answering
unknown achieves 50%). Given correct initial emoji states, deduction accuracy 🎯
is 75% on validation ✅ and 21.9% on test. Given correct expanded states, answer
accuracy 🎯 is 100% on both sets. See measurements in
`runs/semantic-v1/heldout.json`. These results 📊 demonstrate functioning discrete
boundaries, not reliable general reasoning 🧩. The answer head also sees the
original emoji facts, so these measurements do not establish that it causally
uses the added deduction. Revised probes test isolated facts. They do not prove full-state answers use the inferred fact. The
new wording split differs from the original split, so results 📊 across versions
are not a controlled comparison of the architecture alone.

🔬 🧪 🤖 The revised 3,000-step CPU run reaches 100% training 🏋️ accuracy 🎯. Held-out results 📊
remain poor:

| Measurement | Validation (96) | Test (96) |
| --- | ---: | ---: |
| Complete text-to-state parsing | 0% | 6.25% |
| Deduction given correct initial state | 40.6% | 25% |
| Answer given correct expanded state | 100% | 100% |
| End-to-end answer | 38.5% | 56.25% |
| Correct isolated-fact reversal pairs | 16/16 | 16/16 |

🌈 ✨ 🎉 Full results 📊 and example traces are in `runs/semantic/evaluation.json`. The
interventions establish sensitivity to the meaning of an isolated inferred
edge within this tiny vocabulary 🔤; they do not establish reliable deduction or
use of that edge when the original facts are present. More sentence templates
and an attention-based deduction head have not solved compositional
generalization. Both text understanding and deduction still need stronger
training 🏋️ or model adaptation before expanding this into a language model.

✨ 🫪 🌈 🫪 ✨

### Fine-tuning pretrained computation 📚 🧩 💡 🫪

🎯 📊 🔥 `semantic_finetune.py` trains the final SmolLM2 transformer block and final
normalization layer jointly with the parsing, deduction, and answer adapters.
Earlier transformer layers and the atomic symbol embeddings remain frozen.
All three stages share the adapted block. Training 🏋️ caches only outputs of the
frozen prefix; the adapted block is recomputed with gradients every step.
Inference recomputes the full prefix and adapted block for each stage, with only
selected integer emoji IDs passed between stages.

🛠️ ⚙️ 💻 The CPU experiment uses 600 steps with batch size eight on the same data and
splits as the frozen baseline. It selects a checkpoint 💾 using validation ✅ loss
on gold intermediate states every 100 steps, then evaluates the test set once.
The heads train from scratch. The different training 🏋️ budget and checkpoint 💾
selection mean this is a practical comparison, not an isolated causal test of
backbone adaptation. Checkpoints 💾 record the adapted block and normalization
weights; results 📊 record their parameter change to verify actual fine-tuning.

```powershell
.venv\Scripts\python.exe -X utf8 semantic_finetune.py train
.venv\Scripts\python.exe -X utf8 semantic_finetune.py generate "The cat is left of the dog. The dog is left of the bird. Is the cat left of the bird?" --trace
.venv\Scripts\python.exe -m unittest test_model test_reconstruction test_semantic test_semantic_finetune
```

🧠 💬 🚀 Outputs go to `runs/semantic-adapted/`; the frozen baseline
remains in `runs/semantic/`. This adapts one block of the 135M model. The scope remains the four-entity relation prototype 🧪.

🔬 🧪 🤖 The completed run adapts 3,540,672 pretrained parameters and selects step 200
(validation ✅ loss 2.9095). The saved block differs from its pretrained weights
by an L1 sum of 2,299.29. Results 📊 on the same test split are:

✨ 🫪 🌈 🫪 ✨

| Measurement | Frozen baseline | Adapted final block |
| --- | ---: | ---: |
| Complete text-to-state parsing | 6.25% | 0% |
| Deduction given correct initial state | 25% | 12.5% |
| Answer given correct expanded state | 100% | 51.0% |
| End-to-end answer | 56.25% | 50% |
| Correct isolated-fact reversal pairs | 16/16 | 0/16 |

🌈 ✨ 🎉 Validation ✅ end-to-end accuracy 🎯 is also 50%. Validation ✅ loss rises after step
200 while training 🏋️ loss falls, showing overfitting. This checkpoint 💾 does not
improve the model and remains a separate experiment; the frozen baseline is
retained. With fresh heads, a substantially smaller training 🏋️ budget, and only
one adapted block, this result does not establish that broader backbone
fine-tuning would fail. See results 📊 in
`runs/semantic-adapted/evaluation.json`.

## Earlier private-code prototype 📚 🧩 💡 🫪

🎯 📊 🔥 A pretrained language model reads ordinary text. A learned 🌱 bottleneck converts
that input into four discrete emoji tokens. Two learned 🌱 transition rounds each
produce four new discrete emojis. An autoregressive decoder 🔓 answers using only
emojis. It generates no text answer to translate.

```text
Text question
    |
Frozen SmolLM2-135M input encoder (continuous computation)
    |
Four hard emoji selections
    |
Transition -> four hard emoji selections
    |
Transition -> four hard emoji selections
    |
Emoji-only autoregressive answer
```

🛠️ ⚙️ 💻 Only selected emoji tokens cross the reasoning 🧩 boundaries. Transitions cannot
read the text, its encoded features, earlier hidden states, or an attention cache.
The answer decoder 🔓 sees only the final emoji state and previously emitted answer
tokens. Every call reconstructs embeddings from those symbols. The input encoder 🔐
is frozen; the bottleneck, transitions, embeddings, and answer decoder 🔓 are trained.
Input features are standardized using training 🏋️-set statistics saved in the
checkpoint 💾. Validation ✅ examples never contribute to those statistics.
The transition head starts aligned with the emoji embeddings to favor retaining
distinct input symbols; its weights are then freely trained.

🧠 💬 🚀 This is an experimental adaptation of a pretrained language model, not a trained
general-purpose assistant. Computation inside the encoder 🔐 and each transition is
still numerical and continuous. The guarantee concerns information crossing
stage boundaries, not every operation inside a transformer layer.

✨ 🫪 🌈 🫪 ✨

## Run 🔬 🧪 🤖 🫪

🌈 ✨ 🎉 The environment is in `.venv`. Run in PowerShell:

```powershell
.venv\Scripts\python.exe -m unittest -v test_model
.venv\Scripts\python.exe run.py train
.venv\Scripts\python.exe run.py generate "Which animal says woof?"
.venv\Scripts\python.exe run.py generate "Which animal says woof?" --trace
```

📚 🧩 💡 Generation prints only emojis to stdout. `--trace` explicitly
prints diagnostic JSON containing the three emoji states and the answer.
Model loading may print library diagnostics to stderr. CPU is the default;
`run.py --device cuda train` requires a CUDA-enabled PyTorch installation.

🎯 📊 🔥 For a fresh install, use Python 3.12+:

```powershell
uv --cache-dir .uv-cache venv --python 3.12 .venv
uv --cache-dir .uv-cache pip install --python .venv\Scripts\python.exe -r requirements.txt
```

🛠️ ⚙️ 💻 The first run downloads SmolLM2-135M into `.hf-cache`. Training 🏋️ writes
`runs/demo/model.pt` and `runs/demo/evaluation.json`. The checkpoint 💾 contains the
trained emoji model; generation also requires the cached pretrained input encoder 🔐.

✨ 🫪 🌈 🫪 ✨

## Training objective 🧠 💬 🚀 🫪

🔬 🧪 🤖 Only the final answer is supervised. Training prescribes no intermediate emoji traces or text reasoning 🧩 targets. Each answer token, followed by an invisible
end-of-answer token, is predicted autoregressively with cross-entropy loss.

🌈 ✨ 🎉 Training 🏋️ uses a straight-through argmax estimator: the forward pass selects
exactly one emoji per slot; the backward pass uses a softmax gradient approximation.
No sampling noise is added. Fixed weights produce the same choices during training and inference.
Inference uses hard, deterministic argmax selections. This biased approximation may fail to optimize. End/start control tokens are used by the answer
decoder 🔓 only; intermediate states contain exclusively allowlisted emojis.

📚 🧩 💡 The internal alphabet is in `model.py`. Each full emoji is a single token,
including variation selectors and composed keycap emojis. The text tokenizer uses a separate alphabet. Learned 🌱 internal meanings do
not have to match conventional emoji meanings.

## Data and evaluation 🎯 📊 🔥 🫪

🛠️ ⚙️ 💻 The included 48 training 🏋️ questions and 15 distinct validation ✅ questions are a
small, hand-written wiring check covering facts, yes/no questions, arithmetic,
directions, and ordered multi-emoji answers. This dataset is too small for general language understanding. Validation ✅ includes close paraphrases.

✨ 🫪 🌈 🫪 ✨

🧠 💬 🚀 Add your own UTF-8 JSONL files, with one example per line:

```json
{"prompt":"Can a fish breathe underwater? Answer yes or no.","answer":"✅"}
{"prompt":"Name the animal that barks, then the one that meows.","answer":"🐶🐱"}
```

```powershell
.venv\Scripts\python.exe run.py train --data data/train.jsonl --validation data/validation.jsonl --steps 1000
```

🔬 🧪 🤖 Answers must use the allowlisted emojis and fit within 15 tokens plus the end
token. Unsupported symbols are rejected. Input text is never silently truncated.
Training 🏋️ and validation ✅ prompts must be disjoint. The answer never enters the
input encoder 🔐. For a serious experiment, use a much larger dataset and hold out
task families and compositions, not just wording.

🌈 ✨ 🎉 Evaluation records deterministic exact answer accuracy 🎯, answer loss, and a state
intervention: replace each example's final emoji state with another example's
state. A loss increase shows sensitivity to the states; it does not establish
multi-step reasoning 🧩. Several rounds can still learn a redundant code or collapse.
Traces expose each choice. Readable symbols do not prove human-readable learned 🌱 reasoning 🧩.

📚 🧩 💡 The included checkpoint 💾 was trained for 2,000 updates with seed 7:

✨ 🫪 🌈 🫪 ✨

| Measurement | Result |
| --- | --- |
| Training exact answers | 28/48 (58.3%) |
| Validation exact answers | 6/15 (40.0%) |
| Validation answer loss | 0.782 |
| Validation loss with swapped final states | 2.832 |

🎯 📊 🔥 These are development validation ✅ results 📊, not an untouched test benchmark.
Performance remains weak: the model answers the meowing-pet question with a dog,
confuses several numbers and directions, and misses the multi-emoji validation ✅
answers. Some transitions repeat their input state. This run verifies a trainable
discrete architecture, not general language competence or useful iterative reasoning 🧩.
Use `--steps 2000` to run the same training 🏋️ length.

🛠️ ⚙️ 💻 Architectural tests check hard states in training 🏋️ and inference, the inputs at
each transition boundary, gradients from the answer into the quantizers, causal
answer masking, composed emoji tokens, and deterministic emoji-only generation.

## Self-supervised reconstruction experiment 🧠 💬 🚀 🫪

🔬 🧪 🤖 `reconstruction.py` is a separate text autoencoder experiment. It trains directly
on passages from [Project Gutenberg ebook 11](https://www.gutenberg.org/ebooks/11).
Each passage is both input and reconstruction target. No question or intermediate emoji labels are needed. The downloaded source is
`data/alice.txt`; source attribution and its SHA-256 are saved in
`data/reconstruction/metadata.json`.

```powershell
.venv\Scripts\python.exe reconstruction.py prepare
.venv\Scripts\python.exe reconstruction.py compare
.venv\Scripts\python.exe reconstruction.py reconstruct "Alice was beginning to get very tired."
.venv\Scripts\python.exe -m unittest -v test_model test_reconstruction
```

🌈 ✨ 🎉 The experiment uses 2,048 training 🏋️ passages from chapters 1-9, 256 validation ✅
passages from chapter 10, and 256 test passages from chapters 11-12. Passages are
12-48 UTF-8 bytes, whitespace is normalized, and duplicates across splits are
removed before sampling. Very short leftover chunks are discarded. One small book cannot represent general language.

✨ 🫪 🌈 🫪 ✨

📚 🧩 💡 A small trainable byte transformer reads each passage. Fixed-position average
pooling groups its features into 4, 16, 32, or 48 regions, each quantized to an emoji. The reconstruction
decoder 🔓 receives only re-embedded selected emojis plus fixed slot positions;
source features, source masks, and source lengths never reach it. Unlike the
question-answer prototype 🧪, this probe trains its encoder 🔐 from scratch and does
not use SmolLM2. Its text output measures information preservation. The final interface still answers in emojis.

🎯 📊 🔥 Each discrete model has a matched continuous baseline with identical parameters,
initial weights, batches, and optimizer schedule. The baseline uses soft mixtures
of the same codebook instead of hard choices. Both learn with teacher-forced
autoregressive reconstruction. The current comparison runs all eight models for 2,000 updates
with seed 7. Checkpoints 💾 and full evaluation examples are saved in
`runs/reconstruction-positional/`; `comparison.json` contains the measurements.
Use `compare --slots 48` to train just the largest matched pair.
The default reconstruction checkpoint 💾 is now `emoji-48.pt`.

🛠️ ⚙️ 💻 Earlier attention-pooling results 📊, at 600 updates, remain in `runs/reconstruction/`
for reference. These checkpoints 💾 use the removed attention-pooling architecture.
Their results 📊 on the 256 held-out test passages were (lower loss is better):

| State slots | Emoji loss | Continuous loss | Mean distinct emojis per passage |
| --- | --- | --- | --- |
| 4 | 2.214 | 2.145 | 1.03 |
| 16 | 2.215 | 2.157 | 1.13 |
| 32 | 2.215 | 2.155 | 1.32 |

🧠 💬 🚀 Loss is teacher-forced negative log likelihood in nats per prediction, including
the end token. It does not measure free reconstruction. All six earlier models
had **0/16 exact greedy reconstructions** on the fixed test sample. Generation
starts with only a start token and the bottleneck state, with no original text
prefix supplied. The JSON also records free-generation byte accuracy 🎯, emoji
usage, and loss when final states are shuffled between passages.

✨ 🫪 🌈 🫪 ✨

🔬 🧪 🤖 That run failed to demonstrate language compression. In the 32-slot
emoji model, 69.1% of test passages selected the same emoji in every slot. Capacity grew, but learned 🌱 states mostly repeated symbols.
Continuous baselines also reconstructed poorly. Discretization is not the only issue. These results 📊 motivated the positional pooling change.
These are single-seed, short-training 🏋️ results 📊; no architecture was tuned against
the test outcomes.

🌈 ✨ 🎉 The additional tests verify the hard forward states, sole codebook connection
into the decoder 🔓, causal masking, gradients into the source encoder 🔐, matched
baseline parameters, and UTF-8 handling. Together with the original model tests,
there are seventeen passing checks, including independence from other passages'
padding lengths.

### Positional pooling results 📚 🧩 💡 🫪

🎯 📊 🔥 Each slot now pools a fixed region of a 48-byte padded input. Encoder 🔐 outputs at
padding positions are zeroed before pooling, and padding is independent of the
other passages in the batch. Token and positional embeddings start at matching
scales so positions can influence encoding and decoding. The discrete decoder 🔓
still receives only hard emoji choices through the codebook.

🛠️ ⚙️ 💻 The architecture was checked on validation ✅ passages first. The final matched
pairs used identical initial weights, training 🏋️ batches, and 2,000 updates:

✨ 🫪 🌈 🫪 ✨

| Slots | Emoji loss | Continuous loss | Emoji generated byte accuracy | Continuous generated byte accuracy |
| --- | --- | --- | --- | --- |
| 4 | 1.834 | 1.564 | 11.7% | 15.1% |
| 16 | 1.443 | 0.154 | 22.4% | 94.0% |
| 32 | 1.007 | 0.095 | 44.8% | 96.6% |
| 48 | 0.461 | 0.018 | 81.2% | 99.1% |

🧠 💬 🚀 Loss uses all 256 test passages. Generated byte accuracy 🎯 and exact reconstruction
use the same fixed 16-passage sample as before. All discrete models still have
0/16 exact reconstructions; the continuous 16-slot, 32-slot, and 48-slot models achieve
4/16, 8/16, and 12/16 respectively. The 32-slot discrete model averages 11.73 distinct
emojis per passage, with no all-identical states on the test set. Its shuffled
state loss rises from 1.007 to 4.227, showing dependence on the encoded input.

🔬 🧪 🤖 This shows a working baseline and a capacity effect. It does not prove emoji reasoning 🧩. The discrete
32-slot model uses only 15 of the 69 available emojis across test passages and
still loses substantial information. Improving hard-code learning and codebook
usage is the next investigation. Training 🏋️ budgets differ between runs. Pooling alone cannot explain the improvement. The same test chapters have been reused across iterations; treat
these as development results 📊 rather than an independent final benchmark.

### Finite scalar quantization investigation 🌈 ✨ 🎉 🫪

📚 🧩 💡 The opt-in scalar method uses `vector-quantize-pytorch`'s FSQ implementation,
with two coordinates each rounded to eight levels. Its 64 fixed grid points map
to the first 64 emojis in the existing alphabet. Each selected point is projected
through a small nonlinear network into the decoder 🔓's memory 🧠. This is a function
only of the chosen code and global learned 🌱 weights: no original continuous
features bypass the quantizer. Tests verify exact recovery of every transmitted
state from its integer code ID during training 🏋️ and inference, and gradients
through the library's straight-through estimator.

```powershell
.venv\Scripts\python.exe reconstruction.py compare --slots 48 --quantization scalar --validation-only --output runs/reconstruction-fsq-48
.venv\Scripts\python.exe reconstruction.py reconstruct "Alice was beginning to get very tired." --checkpoint runs/reconstruction-fsq-48/fsq-48.pt
```

🎯 📊 🔥 `--validation-only` saves validation ✅ results 📊 without evaluating test passages.
The final 48-slot checkpoints 💾 were evaluated on the test passages after validation ✅
selection; those additional measurements are in their saved `comparison.json`.

✨ 🫪 🌈 🫪 ✨

🛠️ ⚙️ 💻 At 32 slots, the initial linear scalar-to-memory 🧠 mapping recovered 10.1% of
generated validation ✅ bytes. A nonlinear mapping raised that to 19.4% and used
all 64 codes, but remained worse than the prior categorical model's 41.9%.
Low code usage cannot fully explain poor copying.
These were validation ✅-only trials; their reports are in `runs/reconstruction-fsq/`
and `runs/reconstruction-fsq-mlp/`.

🧠 💬 🚀 At 48 slots there is no pooling of adjacent byte positions. All methods below
use the same training 🏋️ passages, seed 7, and 2,000 updates:

| Method | Test loss | Generated byte accuracy | Exact passages | Codes used |
| --- | --- | --- | --- | --- |
| Categorical, 48 slots | 0.461 | 81.2% | 0/16 | 21/69 |
| Scalar, 48 slots | 0.492 | 80.6% | 0/16 | 63/64 |
| Continuous, 48 slots | 0.018 | 99.1% | 12/16 | Not discrete |

🔬 🧪 🤖 The scalar checkpoint 💾 and final results 📊 are in `runs/reconstruction-fsq-48/`.
The 48-slot categorical and continuous controls are in
`runs/reconstruction-control-48/`, and copied into the active positional comparison.
The categorical model remains the recommended discrete method. The improvement
from 32 to 48 categorical slots is a capacity change, not a scalar-quantization
improvement. FSQ improves code utilization but does not improve reconstruction
in these experiments. It also uses a slightly smaller alphabet and a different
decoder 🔓 projection, so this is not a perfectly isolated quantizer comparison.
At 48 slots FSQ can transmit at most 288 bits versus about 293 for categorical
quantization; both exceed the corresponding 32-slot capacities.

🌈 ✨ 🎉 Next, recover exact characters and wording from hard codes. This probes information preservation. It does not prove learned 🌱 reasoning 🧩 or general language competence.

✨ 🫪 🌈 🫪 ✨

📚 🧩 💡 References for this method:

- 🎯 📊 🔥 [Finite Scalar Quantization: VQ-VAE Made Simple](https://arxiv.org/abs/2309.15505)
- [vector-quantize-pytorch implementation](https://github.com/lucidrains/vector-quantize-pytorch) 🛠️ ⚙️ 💻

## References 🧠 💬 🚀 🫪

- 🔬 🧪 🤖 [SmolLM2-135M model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
- [Transformers AutoModel](https://huggingface.co/docs/transformers/en/model_doc/auto) 🌈 ✨ 🎉
- [PyTorch hard Gumbel-softmax](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.gumbel_softmax.html) 📚 🧩 💡
  documents the same straight-through gradient trick; this prototype 🧪 uses argmax
  without Gumbel sampling.

### 🧩 Ordered-content objective trial 🏋️🔍

`objective-v3` trains ordered contextual reconstruction and masked English
reconstruction with reversed/empty-state contrasts. Ambiguous association anchors
are skipped. A fixed evaluation-only set checks answer concepts and participant
order. No additional teacher-generated Q&A is used. 📚🫪

Real states now beat empty states in a small reconstruction audit, but reversed
states still do slightly better. Answer checks remain **2/12**, including **0/4**
participant-order checks. Reliable answering and relational meaning remain unproven.
**75 tests pass.** The English decoder and training projections stay out of runtime.
[Results](data/emoji-grounded/objective-result.json). 🧪🚧

### 🫪 Five-hour standalone development run ⏱️🏋️

`emoji_general_run.py` runs for a persisted five-hour deadline, with atomic training
snapshots and periodic validation. `emoji_general_chat.py` uses the resulting
checkpoint for independent questions. No English answer is generated at runtime. 💬🧠

🌐 The downloaded official Emoji 18.0 file contains **3,972** fully qualified
sequences and components. This run includes all of them, including 🔚 and ▶️ as
ordinary symbols. New symbols use their literal Unicode names; the original
one-time AI association map remains unchanged. Catalog support and learned
meaning recall are measured separately. 🔍📋

📚 Training adds premade [Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
examples, retaining supplied context when it fits. This dataset is **CC-BY-SA-3.0**;
its source and derived data retain that attribution. UltraChat remains MIT.
No new AI-generated Q&A is used. Exact prompts are deduplicated; paraphrase overlap
is not exhaustively checked. The fixed answer checks never select checkpoints. 🧪🚧

💾 Run status, source hashes, checkpoints, coverage, and final examples are saved
under `runs/emoji-general/five-hour/`. Results are pending while the worker runs.
**78 tests pass.** General-chat capability is not established yet. 🫪⏳

🔧 During the five-hour run, validation samples revealed one-symbol topic replies.
Training targets now retain up to six literal/contextual answer codes. Definition
replay is reduced to 5%, and mismatched question states provide a contrastive loss.
The prior checkpoint is retained; validation selection restarts because targets
changed. The original deadline stays fixed. This correction is not yet evidence
of general answering capability. 🧪🚧

### 📊 Standalone run result 🫪🚧

The selected checkpoint passes **5/12** fixed concept checks and **1/4** role-order
checks. All **3,972** official sequences/components are represented. Training-map
literal recall is **3,918/3,972 (98.6%)**; exact supplied-state first-output recall
is **3,517/3,972 (88.5%)**. These are map recall checks, not general-chat accuracy. 🔍

Held-out replies still omit requested information or give unrelated symbols.
Basic general chatting was **not demonstrated**. The model remains experimental.
The logs show progress through about **3h20m**, followed by a large execution/clock
gap; total wall time is **18h37m**. Five hours of continuous training cannot be
verified. [Final audit](data/emoji-grounded/five-hour-result.json). 🧪⏱️

### 🧠 Pretrained direct emoji decoder pilot 🫪🏋️

`emoji_pretrained_model.py` adapts the cached SmolLM2-360M-Instruct transformer
with PEFT LoRA on query/value projections. It consumes only committed emoji IDs
and emoji prefix IDs. Frozen meaning vectors initialize the input lookup;
a new output head predicts 3,972 emoji classes plus an end marker. Each generation
step rebuilds from IDs with no continuous cache. No English answer is generated
or translated at runtime. Numerical activations inside the transformer remain
continuous. 🔢🧩

📚 The existing 9,016 premade training pairs supply the same answer-state targets.
The encoder stays fixed so this trial isolates the reply architecture. The base
weights remain frozen; 5,561,733 adapter/head parameters train. Checkpoints contain
adapters and heads, referencing the pinned cached backbone rather than copying it.
No additional AI-generated Q&A is used. 💾✅

🧪 The initial **150-step** pilot passes **0/12** fixed checks. It produces frequent
symbols or ends early. This is much shorter training than the earlier experiment
and does not establish the architecture's potential. **80 tests pass.** It is not
promoted. [Pilot results](data/emoji-grounded/pretrained-pilot-result.json). 🚧📊

```powershell
.venv\Scripts\python.exe -X utf8 emoji_pretrained_fit.py --name pilot --steps 150
.venv\Scripts\python.exe -X utf8 emoji_pretrained_chat.py --name pilot
```

🏋️ The unattended `aligned-v1` experiment runs **1,000 vocabulary-alignment**
updates, covering all 3,972 symbols, followed by **2,000 premade Q&A** updates.
Half of training prefix positions may use the model's predictions. Runtime
outputs remain direct emoji IDs. Results are pending; this schedule does not
establish correct meanings or general answering capability. 🫪⏳

📉 `aligned-v1` completed 1,000 meaning-alignment and 2,000 answer updates in
about **92 minutes**. It passes **0/12** fixed checks versus the prior model's 5/12.
Only **3/3,972** exact supplied-state first-symbol checks pass, and all ten sampled
held-out replies contain one symbol. This decoder collapsed despite lower loss;
vocabulary support does not establish learned meaning. The existing chat model
is retained. [Final aligned audit](data/emoji-grounded/pretrained-aligned-result.json).
General standalone chatting remains **not demonstrated**. 🧪🚧

### 🔍 Bottleneck diagnostics and pointer trial 🧩

Target first-symbol frequencies are varied: the largest is 247/9,016, so frequency
alone does not explain decoder collapse. Only 20.8% of target codes occur in the
question states. Controlled role and negation pairs produce distinct codes, but
literal content is unreliable: both dog/cat role variants omit the dog, and smoking
allowed already produces prohibition symbols. Distinct codes are not sufficient.
[Input diagnostic](data/emoji-grounded/input-information-diagnostic.json). 🚧📋

`copy-v1` reuses the repository's pointer/generator mixture to preserve supplied
hard emoji IDs. It trains 1,000 definition and 1,000 Q&A updates, with supervision.
Copying a supplied ID proves identity handling, not semantic understanding or
answering. It cannot recover content lost before decoding. New pretrained
checkpoints include pointer parameters; archived non-pointer checkpoints are not
supported by this decoder layout. The existing general chat model is retained.
**81 tests pass.** 🫪🏋️

🧪 `copy-v1` completed in **37 minutes**, selecting step 1,800 of 2,000.
It passes **5/12** fixed checks: equal to the existing general model and above
the pretrained-only trial's 0/12. Supplied-state first-symbol recall is
**3,971/3,972 (99.97%)**. This establishes copying identity, not semantic recall.
Nine of ten sampled held-out replies contain one emoji; the other contains two.
Bees return 🐝 instead of 🍯, and both chase directions return 🐈. The encoder
still loses content and roles. Basic standalone chatting is **not demonstrated**.
The existing chat checkpoint remains active; supervision is paused.
[Copy trial audit](data/emoji-grounded/pretrained-copy-result.json). 📋🚧

### 🐕🐈 Ordered literal input trial 🧪

`anchored-v1` preserves exact core-name matches from the existing meaning map,
in mention order, before filling the eight states with learned IDs. Associations
are excluded: “stay” must not assert 🏨. No question types or new Q&A are added.
The same 1,000 definition and 1,000 answer updates allow comparison with `copy-v1`.
This targets explicit content loss; it does not supply missing facts or establish
role understanding. The active chat model is unchanged. 🫪🧪

📉 The completed trial passes **1/12** fixed checks versus 5/12 for `copy-v1`
and the existing general model. Ordered-role checks remain **0/4**. All 58
matched literal IDs in the held-out set survive; all 30 matching questions retain
their literal prefix order. This is a mechanical guarantee, not semantic recall.
Supplied-state first-symbol recall falls to **2,470/3,972 (62.19%)**. Eight of ten
sampled replies are empty; one has one emoji, and one reaches the nine-symbol cap
without terminating. Input retention alone did not improve answering. 🚧

⏱️ Wall time is **9h24m**, including an approximately **8h50m** gap between log
updates. Continuous training time is unverified. The trial is not promoted,
general chatting remains **not demonstrated**, and supervision is paused.
[Anchored trial audit](data/emoji-grounded/pretrained-anchored-result.json). 📋🫪

```powershell
.venv\Scripts\python.exe -X utf8 emoji_pretrained_fit.py --name anchored-v1 --anchor-input --warmup-steps 1000 --steps 1000
.venv\Scripts\python.exe -X utf8 emoji_anchored_chat.py --name anchored-v1
```

### 🔍 Best-model target audit 📚🫪

The existing answer labels are not reliable meanings. “117” becomes 💯;
“100 yards” becomes 💯🥖; guitar strings become 🧵. Exact core-name retention
is 5,085/5,404, but matching words does not establish their intended meaning.
The target builder also removes repeated IDs and caps content at six symbols,
losing counts and relationships. No new training run is started on these labels.
The best checkpoint stays unchanged. 🚧🔒

Next gate: answer-side, emoji-only information reconstruction on premade data,
with an English decoder used only during training. Test matched, shuffled and
empty states before resuming answering. This is a proposed test, not a completed
improvement. [Target audit](data/emoji-grounded/target-quality-audit.md). 🧪📋

```powershell
.venv\Scripts\python.exe -X utf8 emoji_target_audit.py
```

### 🧩 Blind answer reconstruction gate 🧪

`emoji_reconstruction_fit.py` tests the best encoder's eight ordered emoji states
without deduplication or the six-symbol target cap. A training-only frozen English
backbone and learned adapter reconstruct premade answers. Every answer input
position is an end marker: answer words are never supplied as a prefix. Answer
length and padding remain visible, so this is length-conditioned reconstruction,
not free generation or question answering. 🫪🔢

The isolated `blind-v1` run trains 500 updates with meaning-definition replay.
Validation uses 16 existing examples; the final test uses 32. Real, reversed,
unrelated and empty states receive the same answer lengths. Training snapshots
support resuming; no chat checkpoint is exported or replaced.
No new AI-generated Q&A is used. 📚🔒

📉 The 500-update test completed in **55 minutes**, selecting step 400. On the
same 32 held-out answers, reconstruction loss falls from 17.138 to **9.885**,
but empty states are better (**9.591**). Reversed states score **9.875** and
unrelated states **9.923**. Sampled reconstructions are repeated punctuation
and fragments, not recovered answers. The information gate **fails**. 🚧

All 3,972 definition IDs were replayed across the full run. First-state definition
recall is **3,618/3,972 (91.09%)**, versus **3,633/3,972 (91.47%)** for the original
encoder under the same test. Coverage seen is not recall. This first-position
measure differs from earlier whole-sequence definition audits. No answering run
follows; the best chat model stays unchanged and supervision is paused.
[Reconstruction audit](data/emoji-grounded/blind-reconstruction-result.json). 🫪📋

```powershell
.venv\Scripts\python.exe -X utf8 emoji_reconstruction_fit.py --name blind-v1 --steps 500
```

### 🔬 Continuous control and autonomous diagnostics 🫪

`continuous-v1` replaces hard lookup with soft probabilities over the same frozen
meaning vectors. The encoder, data, decoder, seed and 500-update schedule remain
matched. This isolates hard quantization within the current setup; it is not an
unrestricted continuous encoder. Displayed emoji IDs summarize its argmax choices,
not committed states. It is diagnostic only and cannot replace the emoji runtime.
The completed 500-update control takes **44 minutes**. Held-out losses are real
**9.040**, reversed **9.037**, unrelated **9.052**, and empty **9.727**. Real beats
empty, but corruption barely matters and sampled strings remain fragments.
Useful answer reconstruction is not demonstrated.
[Continuous audit](data/emoji-grounded/continuous-reconstruction-result.json). 📊🚧

Supervision may revise the next experiment from actual results. A failed continuous
control calls for a decoder check; a useful control permits comparison with learned
discrete codes and then anchored emojis. No new generated Q&A, checkpoint promotion
or publishing is authorized by a lower reconstruction loss alone. 📚🔒🧪

`prefix-control-v1` tests standard shifted answer prefixes in the training-only
decoder, retaining the continuous relaxation and 500-update schedule. Full losses
can exploit the English prefix; first-token losses are reported separately because
that position sees only an end marker and the state. Teacher-forced argmax strings
are not free generation. This checks the decoder setup, not general chatting.
The active emoji model remains unchanged. Results are pending. 🔬⏳

### ⚡ Short baseline checks 🫪

On 30 existing test questions, unrelated or empty states change every reply,
but reversing state order changes none. The baseline reply memory has no position
embeddings: cross-attention cannot distinguish permutations of its input states.
A disposable clone learns 16 existing targets exactly in 150 updates (about nine
seconds), so tiny-set optimization works. This is memorization of weak labels,
not semantic answering. Forcing three-symbol replies mostly adds repetitions or
unrelated symbols. The active checkpoint is unchanged. 🔒🧪

[Quick-check report](data/emoji-grounded/quick-checks.md). 📋

⚡ A short positional-memory ablation copies **16/16** trained ordered pairs,
versus **8/16** without positions. On unseen pairs, it reaches only **1/16** versus
0/16. Positions fix permutation invariance; they do not establish general copying
or semantic reasoning. The matched check takes about **14 seconds** of compute,
and the active model remains unchanged.
[Position experiment](data/emoji-grounded/position-check.md). 🫪🧪

### 🚀 Simple pretrained two-call prototype 🫪

`emoji_two_pass.py` uses an unmodified pretrained instruct model and Outlines
with LLGuidance constraints. The first call receives an English question and
generates an emoji sequence. A fresh second call receives only that sequence
and fixed instructions, then generates an emoji answer. No English answer is
generated or translated, and no hidden cache crosses calls. JSON is transport;
interactive output displays emojis. Numerical activations inside each call are
ordinary continuous model computations. 🧠➡️🫪➡️🫪

All **3,972** catalog symbols are allowed, with at most four per stage in this
small prototype. No training or custom encoder is used. The cached 360M model
passes only **1/6** simple fixed checks, taking about **2–6 seconds** per question
after initialization. Every second-stage answer repeats its intermediate state
in this sample. Valid emoji output does not establish reasoning or useful chat.
The prototype works structurally; answer quality remains poor. 🚧📊

```powershell
.venv\Scripts\python.exe -X utf8 emoji_two_pass.py
.venv\Scripts\python.exe -X utf8 emoji_two_pass.py --question "Which animal barks?"
```

The active trained baseline stays unchanged. Earlier decoder experiments are
archived; this simpler route replaces further custom-architecture development.
[Prototype results](data/emoji-grounded/two-pass-result.json). 📋🔒
