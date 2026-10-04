# Conversation state grammar

Each complete Unicode emoji sequence is one atomic token. The full catalogue
contains 3,953 tokens, including joined sequences and skin-tone variants.
`meanings.json` records names; the following patterns define conversation
operators. Meaning depends on position in a pattern, as in an ordinary grammar.

| Pattern | Meaning | Example |
| --- | --- | --- |
| `💖 ENTITY` | User likes ENTITY | `💖🍕` |
| `💔 ENTITY` | User dislikes ENTITY | `💔🥦` |
| `🔄💖 ENTITY` | Replace the earlier positive preference | `🔄💖🍣` |
| `🧠💖❓` / `🧠💔❓` | Recall positive / negative preference | `🧠💖❓` |
| `📌 ENTITY 🎨 COLOR` | Store an entity's color | `📌🚲🎨🔵` |
| `🔄 ENTITY 🎨 COLOR` | Replace an entity's color | `🔄🚲🎨🔴` |
| `❓ ENTITY 🎨` | Ask about that entity's color | `❓🚲🎨` |
| `❓🎨` | Ask about the last mentioned entity's color | `❓🎨` |
| `📌 ENTITY 📍 PLACE` | Store an entity's location | `📌🚲📍🏠` |
| `🔄 ENTITY 📍 PLACE` | Update an entity's location | `🔄🚲📍🏞️` |
| `❓ ENTITY 📍` | Ask where the entity is | `❓🚲📍` |
| `📌 ENTITY 🔢 DIGIT` | Store a quantity from zero through nine | `📌🍎🔢3️⃣` |
| `🔄 ENTITY 🔢 DIGIT` | Replace a stored quantity | `🔄🍎🔢5️⃣` |
| `❓ ENTITY 🔢` | Recall a stored quantity | `❓🍎🔢` |
| `💖 A 💔 B ⚖️❓` | Choose using the stated preferences | `💖🍕💔🥦⚖️❓` |
| `🔢 A ➕ B` | Ask for a small sum | `🔢2️⃣➕3️⃣` |
| `🔢 ENTITY A ➕ B` | Ask for an entity count | `🔢🍎2️⃣➕3️⃣` |
| `🎯 TOPIC` | Request a simple action plan | `🎯📚` |
| `👋`, `🙏`, `😔`, … | Greeting, thanks, sadness, other defined social intents | See `conversation_data.py` |
| `❓` | Unsupported question in this curriculum | Reply `🤔❓` |
| `⁉️` | Request needs clarification | Reply `🤔❓💬` |

Memory stores whole turns as `👤 STATE 🔹 🤖 REPLY 🔹`. `▶️` starts generation
and `🔚` ends it; neither appears in the displayed reply. A reply such as
`💖🍕` recalls a preference; `👍🍕` acknowledges a declaration. Replies are
generated directly by a neural decoder from emoji memory, not from English
assistant answers. Training teaches these conventions; the runtime does not
execute a hand-written fact solver or select from a canned-reply table.

The reliability curriculum mixes color, location, and quantity facts in one
history. User declarations define the intended facts. A conflicting assistant
reply does not change them. The latest user update replaces only its named
attribute. An unknown entity or missing attribute receives `ENTITY ❓`.
Quantity recall returns `ENTITY 🔢 DIGIT`. Color recall returns `ENTITY COLOR`.
Location recall returns `ENTITY 📍 PLACE`.

`conversation_reliability_data.py` defines 16 additional everyday topics with
authored state and reply patterns. These include cleaning, organization,
language learning, writing, repair, coding, debugging, teamwork, focus,
creativity, walking, gravity, computers, the internet, electricity, and
algorithms. The short replies communicate only the documented outline.
They do not express detailed instructions or arbitrary explanations.

These definitions make the intended states readable and testable. They do not
guarantee that the model selects the right state. This curriculum is a limited
semantic language, not a demonstrated encoding of arbitrary human conversation.
Named embeddings and transient numerical activations are still continuous.
Only the selected tokens and persistent conversation memory are discrete.
