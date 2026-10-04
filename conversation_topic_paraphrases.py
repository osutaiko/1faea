"""Generate reviewable user paraphrases for the expanded everyday topics."""

from conversation_paraphrases import generate
from conversation_reliability_data import DIRECTORY, TOPICS


if __name__ == '__main__':
    specs = [(skill, state, reply, training) for skill, state, reply, training, *_ in TOPICS]
    generate(specs, DIRECTORY / 'generated-topic-prompts.jsonl')
