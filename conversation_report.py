"""Summarize the completed CPU development run from saved evaluations."""

from datetime import datetime, timezone
import json

from run import ROOT


def report():
    directory = ROOT / 'runs' / 'conversation-compositional'
    models = ('baseline', 'selected', 'dialogue')
    evaluations = {name: json.loads((directory / name / 'evaluation-test.json').read_text(encoding='utf-8'))
                   for name in models}
    audits = {name: json.loads((directory / name / 'audit.json').read_text(encoding='utf-8')) for name in models}
    attributes = {name: json.loads((directory / name / 'attribute-audit.json').read_text(encoding='utf-8'))
                  for name in models}
    lines = ['# Conversation development results', '',
             'All variants use ordinary text input and directly generate emoji replies through hard emoji states.',
             'Transient neural activations remain continuous; persistent memory contains only emoji IDs.', '',
             '## Controlled test split', '',
             'Each variant was evaluated on the same 2,189 rows with correct prior emoji history.',
             'The split was evaluated iteratively during development; it is not an independent final blind audit.', '',
             '| Checkpoint | Exact state | Exact reply | Correct state and reply | Reply given gold state |',
             '| --- | ---: | ---: | ---: | ---: |']
    for name in models:
        metrics = evaluations[name]['metrics']['all']
        values = [f"{metrics[key]:.1%}" for key in ('meaning', 'reply', 'correct_trace', 'reply_given_gold_state')]
        lines.append('| ' + ' | '.join([name, *values]) + ' |')
    lines.extend(['', '## Development smoke tests', '',
                  'These small diagnostics guided curriculum development and do not establish general quality.', '',
                  '| Checkpoint | Live reply accuracy (17 turns) | Swapped-color accuracy (8 queries) | Mixed-attribute accuracy (8 queries) | Query operator changed reply (4 pairs) |',
                  '| --- | ---: | ---: | ---: | ---: |'])
    for name in models:
        live = audits[name]
        attribute = attributes[name]
        lines.append(f"| {name} | {live['live_reply_accuracy']:.1%} | {live['counterfactual_accuracy']:.1%} | "
                     f"{attribute['exact_reply_accuracy']:.1%} | {attribute['operator_changes_reply']}/4 |")
    lines.extend(['', '## Findings', '',
                  '- Reviewed input augmentation improved cached validation state accuracy from 56.9% to 82.3%.',
                  '- The longer-dialogue pass improved the live smoke test but regressed on swapped-color binding.',
                  '- Changing color/location query operators did not change replies in the mixed-attribute audit.',
                  '- In the selected checkpoint, location recall had 96.9% correct replies and 0% correct query states. Reply accuracy alone hides this failure.',
                  '- Unfamiliar aliases, unsupported questions, held-out arithmetic, and public natural prompts remain unreliable.',
                  '- The intended emoji grammar is readable; learned use of that grammar is not yet dependable.', '',
                  '## Delivered', '',
                  '- Frozen SmolLM2-360M-Instruct body with trained input and reply heads; full 3,953-token emoji vocabulary.',
                  '- 10,634 compositional training rows, 788 reviewed/authored input augmentations, 768 memory rows, and 960 longer-dialogue rows.',
                  '- Interactive chat, explicit state traces, bounded whole-turn emoji memory, and reproducible checkpoint selection.',
                  '- 51 passing tests plus full runtime evaluations and separate semantic intervention audits.', '',
                  'The selected short-memory checkpoint remains the conservative demo. The dialogue checkpoint is a research variant with a measured tradeoff.',
                  'Neither is ready for general use. The next training curriculum must mix attributes and dialogue types, retain earlier memory validation, and enforce the intended meanings through counterfactual examples.', ''])
    (directory / 'development.md').write_text('\n'.join(lines), encoding='utf-8')
    metadata_path = ROOT / 'runs' / 'conversation' / 'development-run.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    finished = datetime.fromisoformat(metadata['finished']) if 'finished' in metadata else datetime.now(timezone.utc)
    minimum = datetime.fromisoformat(metadata['minimum_finish'])
    if finished < minimum:
        raise ValueError('The requested minimum development duration has not elapsed')
    metadata.update(finished=finished.isoformat(), updated=finished.isoformat(),
                    elapsed_seconds=(finished - datetime.fromisoformat(metadata['started'])).total_seconds(),
                    status='completed development run; general conversation remains experimental',
                    tests_passed=51, results={name: evaluation['metrics']['all'] for name, evaluation in evaluations.items()},
                    summary=str((directory / 'development.md').relative_to(ROOT)))
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    report()
