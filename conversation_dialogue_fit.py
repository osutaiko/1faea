"""Adapt emoji memory to longer dialogues and unreliable assistant history."""

import argparse
import json
import random
import shutil

import torch

import conversation
from conversation_composition import COLORS
from conversation_data import IDS, SKILLS, USER, ASSISTANT, SEPARATOR, pack_history, record
from conversation_memory_fit import prepare, train
from emoji_catalog import ALPHABET, ENTITIES
from run import ROOT


def examples(count, seed):
    rng = random.Random(seed)
    reserved = {IDS[symbol] for _, state, reply, *_ in SKILLS for symbol in (*state, *reply)}
    reserved.update(IDS[symbol] for symbol in ('💖', '💔', '🔄', '🧠', '🎨', '📍', '🔢', '➕', '👌'))
    reserved.update(IDS[symbol] for _, symbol in COLORS)
    reserved.update((USER, ASSISTANT, SEPARATOR))
    entities = [ALPHABET[entity] for entity in ENTITIES if entity not in reserved]
    rows = []
    for index in range(count):
        a, b, c = rng.sample(entities, 3)
        history = []
        def turn(state, reply, skill, corrupt=None):
            item = record('', state, reply, skill, pack_history(history))
            rows.append(item)
            history.append((item['state'], [IDS[symbol] for symbol in corrupt] if corrupt else item['reply']))
        if index % 2:
            turn(('👋',), ('👋', '😊'), 'dialogue_greeting')
        turn(('💖', a), ('👍', a), 'dialogue_preference', ('💖', b) if index % 4 == 0 else None)
        turn(('🧠', '💖', '❓'), ('💖', a), 'dialogue_recall')
        turn(('💔', b), ('👌', '🚫', b), 'dialogue_dislike')
        turn(('🧠', '💔', '❓'), ('💔', b), 'dialogue_recall')
        _, social_state, social_reply, *_ = rng.choice(SKILLS[:24])
        turn(social_state, social_reply, 'dialogue_social')
        turn(('🔄', '💖', c), ('👌', '💖', c), 'dialogue_correction', ('💖', a) if index % 4 == 0 else None)
        turn(('🧠', '💖', '❓'), ('💖', c), 'dialogue_updated_recall')
        history = []
        if index % 2:
            turn(social_state, social_reply, 'dialogue_social')
        color_a, color_b = [symbol for _, symbol in rng.sample(COLORS, 2)]
        turn(('📌', a, '🎨', color_a), ('👍', a, color_a), 'dialogue_color_fact')
        turn(('📌', b, '🎨', color_b), ('👍', b, color_b), 'dialogue_color_fact')
        questions = [(a, color_a), (b, color_b)]
        rng.shuffle(questions)
        for entity, color in questions:
            turn(('❓', entity, '🎨'), (entity, color), 'dialogue_color_recall')
        turn(social_state, social_reply, 'dialogue_social')
        turn(('❓', a, '🎨'), (a, color_a), 'dialogue_color_recall')
        turn(('❓', c, '🎨'), (c, '❓'), 'dialogue_unknown_color')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='dialogue')
    parser.add_argument('--steps', type=int, default=3000)
    args = parser.parse_args()
    torch.set_num_threads(2)
    conversation.OUTPUT = ROOT / 'runs' / 'conversation-compositional'
    output = conversation.OUTPUT / args.name
    output.mkdir(exist_ok=True)
    for part in ('encoder', 'decoder'):
        shutil.copyfile(conversation.OUTPUT / 'selected' / f'{part}.pt', output / f'{part}.pt')
    manifest = json.loads((conversation.OUTPUT / 'features.json').read_text(encoding='utf-8'))
    dataset = dict(train=examples(64, 337), validation=examples(16, 347))
    supplemental = prepare(conversation.OUTPUT, manifest['base_model'], dataset, 'dialogue-features')
    train(args, manifest, supplemental)
