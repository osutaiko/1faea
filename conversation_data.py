"""Grounded conversational examples with directly authored emoji states/replies."""

import json
import random

from emoji_catalog import ALPHABET, CATALOG, ENTITIES
from run import ROOT


IDS = {symbol: index for index, symbol in enumerate(ALPHABET)}
USER, ASSISTANT, SEPARATOR = (IDS[symbol] for symbol in ('👤', '🤖', '🔹'))
DIRECTORY = ROOT / 'data' / 'conversation'

# State and reply symbols are authored directly. There is no English assistant
# answer that is generated and subsequently converted into these targets.
SKILLS = (
    ('greeting', ('👋',), ('👋', '😊'),
     ('Hello!', 'Hi there.', 'Hey!', 'Good morning.', 'Good evening.', 'Greetings.', 'Hi friend.'),
     ('Hey, how is it going?', 'Hello there, nice to meet you.'),
     ('Howdy!', 'Yo, anyone here?', 'A very good afternoon to you.')),
    ('thanks', ('🙏',), ('🤝', '😊'),
     ('Thank you.', 'Thanks a lot!', 'I appreciate your help.', 'Many thanks.', 'That was helpful, thanks.'),
     ('Really appreciate it!', 'Thanks for explaining that.'),
     ('Much obliged.', 'You have been a great help.', 'Cheers for that!')),
    ('farewell', ('🚪', '👋'), ('👋', '🌟'),
     ('Goodbye.', 'Bye for now.', 'See you later!', 'I have to go.', 'Talk to you soon.', 'Good night.'),
     ('Catch you later.', 'I am heading out now.'),
     ('Until we meet again.', 'Time for me to sign off.', 'Take care, I am leaving.')),
    ('apology', ('🙇',), ('🤝', '😊'),
     ('Sorry about that.', 'I apologize.', 'My mistake, sorry.', 'Please forgive me.', 'Sorry, I got it wrong.'),
     ('My apologies for the confusion.', 'I am sorry I was rude.'),
     ('That was my fault.', 'I owe you an apology.', 'Pardon my behavior.')),
    ('identity', ('❓', '🤖'), ('🤖', '💬', '😊'),
     ('Who are you?', 'What are you?', 'Are you a robot?', 'Tell me about yourself.', 'What do you do?'),
     ('Are you an AI assistant?', 'Introduce yourself.'),
     ('Who am I talking to?', 'What kind of assistant is this?', 'Describe your role.')),
    ('capabilities', ('❓', '💬'), ('💬', '🧠', '🤝'),
     ('Can you help me?', 'What can you help with?', 'I need some help.', 'What can you do?', 'Can we talk?'),
     ('Could you lend me a hand?', 'I would like your assistance.'),
     ('How can you assist me today?', 'I could use someone to talk to.', 'Is help available here?')),
    ('emoji_format', ('❓', '😊', '💬'), ('😊', '💬', '✅'),
     ('Do you speak in emojis?', 'Can you answer using emojis?', 'Why are your replies emojis?', 'Use only emojis please.'),
     ('Are all your responses emoji symbols?', 'Please communicate with pictures.'),
     ('Is this an emoji-only chatbot?', 'Could we converse entirely through emoji?', 'Explain your output format.')),
    ('sadness', ('😔',), ('🫂', '💛', '👂'),
     ('I feel sad.', 'I am feeling down today.', 'I have been crying.', 'I am unhappy.', 'Today has been really hard.'),
     ('Everything feels gloomy right now.', 'I am having a rough day.'),
     ('I feel blue.', 'My spirits are low.', 'I cannot stop feeling miserable.')),
    ('loneliness', ('😔', '👤'), ('🫂', '🤝', '💬'),
     ('I feel lonely.', 'I have nobody to talk to.', 'I am alone and sad.', 'I miss having company.', 'I feel isolated.'),
     ('It seems like nobody is around for me.', 'I wish I had someone to talk with.'),
     ('I am longing for companionship.', 'Being on my own is getting to me.', 'I feel disconnected from everyone.')),
    ('anxiety', ('😟',), ('🫂', '🌬️', '🧘'),
     ('I feel anxious.', 'I am worried.', 'I feel nervous today.', 'I am stressed out.', 'I cannot relax.'),
     ('My worries keep racing through my head.', 'I am feeling tense.'),
     ('I am on edge.', 'My mind will not settle down.', 'I have butterflies in my stomach.')),
    ('anger', ('😡',), ('🌬️', '🧘', '💬'),
     ('I am angry.', 'I am so frustrated.', 'This makes me furious.', 'I am upset with everyone.', 'I feel mad.'),
     ('I could scream with frustration.', 'I am losing my temper.'),
     ('I am seeing red.', 'I am absolutely fuming.', 'My patience has run out.')),
    ('happiness', ('😄',), ('🎉', '😄', '🙌'),
     ('I feel happy!', 'I am excited!', 'Today was wonderful.', 'I have great news!', 'I am delighted.'),
     ('Things are going really well.', 'I am in a great mood.'),
     ('I am over the moon.', 'I could not be happier.', 'I am walking on sunshine.')),
    ('tiredness', ('😴',), ('🛌', '💤'),
     ('I am tired.', 'I feel sleepy.', 'I need some rest.', 'I am exhausted.', 'I barely slept last night.'),
     ('I cannot keep my eyes open.', 'I am worn out.'),
     ('I am running on empty.', 'I could fall asleep standing up.', 'I need to recharge.')),
    ('hunger', ('😋', '🍽️'), ('🍽️', '🥗', '😋'),
     ('I am hungry.', 'I need something to eat.', 'My stomach is growling.', 'It is time for lunch.', 'I want some food.'),
     ('I have not eaten all day.', 'I could really use a meal.'),
     ('I am starving.', 'Any ideas for a bite to eat?', 'I am feeling peckish.')),
    ('thirst', ('🥤',), ('💧', '🥤'),
     ('I am thirsty.', 'I need a drink of water.', 'I want some water.', 'My mouth feels dry.', 'I have not had enough to drink.'),
     ('Could I get a glass of water?', 'I need to hydrate.'),
     ('I am parched.', 'I could use something refreshing to drink.', 'Time to quench my thirst.')),
    ('boredom', ('🥱',), ('🎮', '📚', '🚶'),
     ('I am bored.', 'There is nothing to do.', 'I need something fun to do.', 'I want an activity.', 'I am feeling bored today.'),
     ('How can I pass the time?', 'I would like some entertainment.'),
     ('I am twiddling my thumbs.', 'Give me an idea to beat boredom.', 'I have an idle afternoon to fill.')),
    ('celebrate', ('🎉',), ('🎉', '🥳', '👏'),
     ('I passed my exam!', 'I got promoted!', 'I did it!', 'I won the competition!', 'I reached my goal.'),
     ('I finally finished my project!', 'I accomplished something big today.'),
     ('My hard work paid off.', 'I have an achievement to celebrate.', 'I succeeded after many attempts.')),
    ('birthday', ('🎂',), ('🎂', '🎉', '🎁'),
     ('It is my birthday!', 'Today is my birthday.', 'Wish me a happy birthday.', 'I am celebrating my birthday.'),
     ('Another year older today!', 'Can you celebrate my birthday with me?'),
     ('I am turning a year older.', 'Today marks the day I was born.', 'Send birthday wishes.')),
    ('encouragement', ('🎯', '💪'), ('💪', '🌟', '🙌'),
     ('Please encourage me.', 'I need motivation.', 'I want to keep trying.', 'Can you cheer me on?', 'I need some encouragement.'),
     ('Help me feel motivated to continue.', 'I could use a pep talk.'),
     ('Give me a boost of confidence.', 'Remind me I can persevere.', 'I need a little push to get going.')),
    ('study', ('🎯', '📚'), ('📚', '📝', '⏱️', '☕'),
     ('How should I study?', 'Help me plan a study session.', 'I want to prepare for an exam.', 'I need to learn effectively.', 'Give me study advice.'),
     ('How can I organize my revision?', 'Suggest a study routine.'),
     ('What is a sensible way to review my lessons?', 'Help me get ready for a test.', 'I want a learning schedule.')),
    ('exercise', ('🎯', '🏃'), ('🚶', '🏃', '💧', '🛌'),
     ('I want to exercise.', 'Suggest a workout routine.', 'How can I get more active?', 'I want to start working out.', 'Give me fitness ideas.'),
     ('Help me build an exercise habit.', 'What can I do to move more?'),
     ('I want to improve my physical activity.', 'How should a beginner approach working out?', 'Give me a simple movement plan.')),
    ('sleep_plan', ('🎯', '😴'), ('📵', '🌙', '🛌', '💤'),
     ('How can I sleep better?', 'Give me a bedtime routine.', 'I want to improve my sleep.', 'Help me get to sleep.', 'I need a good sleep routine.'),
     ('How should I wind down before bed?', 'Suggest a calmer evening routine.'),
     ('What habits might help me rest at night?', 'Help me settle down for bedtime.', 'I want better nighttime rest.')),
    ('relax', ('🎯', '🧘'), ('🌬️', '🧘', '🎵'),
     ('Help me relax.', 'How can I calm down?', 'I want to unwind.', 'Suggest a relaxing activity.', 'I need to take a break.'),
     ('How can I find a moment of calm?', 'Give me a way to de-stress.'),
     ('I need to decompress.', 'Suggest something soothing.', 'Help me slow things down.')),
    ('travel_plan', ('🎯', '✈️'), ('🗺️', '🎒', '🎫', '✈️'),
     ('Help me plan a trip.', 'I want to go traveling.', 'How should I prepare for travel?', 'I am planning a vacation.', 'Give me travel preparation steps.'),
     ('What should I sort out before a journey?', 'Help me organize a holiday.'),
     ('I would like a checklist for going away.', 'How do I get ready to travel?', 'Suggest steps for an upcoming trip.')),
    ('packing', ('🎯', '🎒'), ('👕', '🪥', '📱', '🎫', '🎒'),
     ('What should I pack?', 'Help me pack a bag.', 'I need a packing list.', 'What goes in my suitcase?', 'I am packing for a trip.'),
     ('What essentials should I take with me?', 'Suggest items for my travel bag.'),
     ('Make a list of things to bring on holiday.', 'What should go into my luggage?', 'Help me remember the basics to pack.')),
    ('cooking', ('🎯', '🍳'), ('🥕', '🔪', '🍳', '🍽️'),
     ('I want to cook.', 'Help me prepare a meal.', 'What are basic cooking steps?', 'Give me an easy cooking plan.', 'How do I make dinner?'),
     ('I would like to prepare some food.', 'Suggest a simple meal preparation routine.'),
     ('How can a beginner get started in the kitchen?', 'Walk me through preparing dinner.', 'I want an uncomplicated cooking sequence.')),
    ('gardening', ('🎯', '🌱'), ('🌱', '🪴', '💧', '☀️'),
     ('How do I grow a plant?', 'I want to start gardening.', 'Help me care for a plant.', 'What do plants need?', 'Give me gardening basics.'),
     ('How should I look after a seedling?', 'I would like to grow something.'),
     ('What helps a young plant thrive?', 'I am a beginner with plants.', 'Suggest a plant care routine.')),
    ('rain', ('☔',), ('☔', '🧥'),
     ('It is raining outside.', 'What should I take in the rain?', 'I have to go out in wet weather.', 'Rain is falling.', 'I am going outside while it rains.'),
     ('How should I prepare for rainy weather?', 'It looks wet outdoors.'),
     ('There is a downpour outside.', 'What would help on a rainy walk?', 'I am heading out into a shower.')),
    ('sun', ('☀️',), ('🧢', '🕶️', '💧'),
     ('It is sunny outside.', 'What should I take on a sunny day?', 'It is hot and sunny.', 'I am going out in sunshine.'),
     ('How should I prepare for a bright day?', 'The sun is shining strongly.'),
     ('I am heading out under a blazing sun.', 'What is useful on a hot sunny walk?', 'It is a very bright afternoon.')),
    ('music', ('❓', '🎵'), ('🎵', '🎧', '😊'),
     ('I want to listen to music.', 'Suggest something musical.', 'I love listening to songs.', 'Can we talk about music?', 'I would like some music.'),
     ('I feel like putting on some tunes.', 'What is a nice way to enjoy music?'),
     ('I am in the mood for melodies.', 'Help me enjoy some songs.', 'I could use a soundtrack for my day.')),
    ('reading', ('❓', '📚'), ('📚', '☕', '😊'),
     ('I want to read a book.', 'Suggest a reading activity.', 'I love books.', 'Can we talk about reading?', 'I would like to spend time reading.'),
     ('I feel like getting lost in a story.', 'Help me find time to read.'),
     ('I am in the mood for a good novel.', 'I want a quiet afternoon with a book.', 'Suggest a cozy reading session.')),
    ('gaming', ('❓', '🎮'), ('🎮', '😊', '⏱️'),
     ('I want to play a game.', 'Suggest a gaming activity.', 'I like video games.', 'Can we talk about games?', 'I feel like gaming.'),
     ('I would like some time playing games.', 'How can I enjoy a game break?'),
     ('I am in the mood for a controller and a game.', 'I want a little gaming session.', 'Help me plan some time for games.')),
    ('art', ('❓', '🎨'), ('🎨', '🖌️', '😊'),
     ('I want to make art.', 'Suggest a creative activity.', 'I like painting.', 'Can we talk about art?', 'I want to draw something.'),
     ('I feel like doing something artistic.', 'Help me get creative.'),
     ('I want to express myself with colors.', 'Suggest something to do with a paintbrush.', 'I could use a creative outlet.')),
    ('cats', ('❓', '🐱', '🗣️'), ('🐱', '🗣️', '🎵'),
     ('What animal says meow?', 'Which animal meows?', 'What sound does a cat make?', 'Tell me about the sound cats make.'),
     ('Which pet makes a meowing sound?', 'Who goes meow?'),
     ('What creature is known for meowing?', 'Identify the animal that purrs and meows.', 'How do cats vocalize?')),
    ('bees', ('❓', '🐝', '🍯'), ('🐝', '➡️', '🍯'),
     ('What do bees produce?', 'Where does honey come from?', 'Which animal makes honey?', 'Tell me about bees and honey.'),
     ('What sweet food do bees make?', 'Who produces honey?'),
     ('Which insect is responsible for honey?', 'How are bees related to honey?', 'Name the creature that makes honey.')),
    ('fish_home', ('❓', '🐟', '🏠'), ('🐟', '🌊'),
     ('Where do fish live?', 'What is a fish habitat?', 'Do fish live in water?', 'Tell me where fish can live.'),
     ('What environment do fish need?', 'Where would you find a fish living?'),
     ('What sort of home does a fish have?', 'Describe the natural surroundings of fish.', 'Where are fish usually found?')),
    ('bird_flight', ('❓', '🐦', '🪽'), ('🐦', '🪽', '☁️'),
     ('How do birds fly?', 'What do birds use to fly?', 'Tell me about bird wings.', 'Why do birds have wings?'),
     ('What helps a bird get into the air?', 'How can a bird move through the sky?'),
     ('What body part allows many birds to fly?', 'Explain birds taking flight.', 'How are wings useful to birds?')),
    ('photosynthesis', ('❓', '🌱', '☀️'), ('🌱', '☀️', '💧', '➡️', '🌿'),
     ('What is photosynthesis?', 'How do plants use sunlight?', 'How do plants grow using light?', 'Tell me about plants and sunlight.'),
     ('Why do plants need sunshine?', 'How does sunlight help plants grow?'),
     ('Explain how plants get energy from the sun.', 'What do plants do with light and water?', 'Describe photosynthesis using emojis.')),
    ('water_cycle', ('❓', '💧', '🔄'), ('🌊', '☀️', '☁️', '🌧️', '🌊'),
     ('What is the water cycle?', 'Explain the rain cycle.', 'How does water move through nature?', 'Tell me about evaporation and rain.'),
     ('How does water cycle around the earth?', 'Why does water return as rainfall?'),
     ('Describe evaporation, clouds, and precipitation.', 'Show the journey from ocean water to rain.', 'What happens during the hydrologic cycle?')),
    ('recycling', ('❓', '♻️'), ('🗑️', '♻️', '🌍'),
     ('What is recycling?', 'Why should we recycle?', 'Tell me how recycling helps.', 'I want to recycle more.'),
     ('How can reusing materials help the planet?', 'What is the point of recycling waste?'),
     ('Explain turning waste into useful material.', 'How does recycling support the environment?', 'I want to reduce what I throw away.')),
    ('unsupported', ('❓',), ('🤔', '❓'),
     ('Tell me the exact current stock price.', 'What is my password?', 'What is happening in live news?', 'Who will win tomorrow?', 'Give me private account information.', 'What is my bank balance?', 'Explain an advanced quantum field calculation.'),
     ('What will the lottery numbers be?', 'Access my private files.'),
     ('Predict the precise outcome of next week.', 'What secrets does my neighbor have?', 'Look up the live price of this asset.')),
    ('clarification', ('⁉️',), ('🤔', '❓', '💬'),
     ('Do that thing.', 'Help with it.', 'Fix that.', 'You know what I mean.', 'What about that one?', 'Make it better.'),
     ('Can you do the stuff?', 'I want the other thing.'),
     ('Sort out whatever needs sorting.', 'It is that thing again.', 'Handle it for me.')),
)


def ids(symbols):
    return [IDS[symbol] for symbol in symbols]


def record(text, state, reply, skill, history=()):
    return dict(text=text, state=ids(state), reply=ids(reply), skill=skill, history=list(history))


def pack_history(turns):
    return [token for state, reply in turns for token in
            [USER, *state, SEPARATOR, ASSISTANT, *reply, SEPARATOR]]


def splits():
    result = {name: [] for name in ('train', 'validation', 'test')}
    for skill, state, reply, training, validation, test in SKILLS:
        for name, phrases in [('train', training), ('validation', validation), ('test', test)]:
            for text in phrases:
                result[name].append(record(text, state, reply, skill))
                if name == 'train':
                    # Surface variations preserve semantics and do not introduce
                    # train/test paraphrases from the evaluation sets.
                    for variant in (text.lower(), text.upper(), 'Please, ' + text, text + ' Can you respond?'):
                        result[name].append(record(variant, state, reply, skill))
    reserved = {token for _, state, reply, *_ in SKILLS for token in ids((*state, *reply))}
    reserved.update((USER, ASSISTANT, SEPARATOR))
    entities = [entity for entity in ENTITIES if entity not in reserved]
    forms = dict(
        train=[('I like {a}.', ('💖',)), ('I love {a}.', ('💖',)), ('My favorite is {a}.', ('💖',)),
               ('I dislike {a}.', ('💔',)), ('I do not like {a}.', ('💔',)), ('I hate {a}.', ('💔',))],
        validation=[('I am fond of {a}.', ('💖',)), ('I am not keen on {a}.', ('💔',))],
        test=[('I really enjoy {a}.', ('💖',)), ('I cannot stand {a}.', ('💔',))])
    for name, count, seed in [('train', 1536, 101), ('validation', 128, 103), ('test', 256, 107)]:
        rng = random.Random(seed)
        for index in range(count):
            a, b = rng.sample(entities, 2)
            label = CATALOG[a]['unicode_name'] if index % 2 else ALPHABET[a]
            form, mood = rng.choice(forms[name])
            state = (*mood, ALPHABET[a])
            reply = ('👍', ALPHABET[a]) if mood == ('💖',) else ('👌', '🚫', ALPHABET[a])
            first = record(form.format(a=label), state, reply, 'preference')
            result[name].append(first)
            history = pack_history([(first['state'], first['reply'])])
            question = dict(train='What do I like?' if mood == ('💖',) else 'What do I dislike?',
                            validation='Can you recall my preference?' if mood == ('💖',) else 'Can you recall what I dislike?',
                            test='What was the thing I enjoyed?' if mood == ('💖',) else 'Which thing did I say I hated?')[name]
            result[name].append(record(question, ('🧠', *mood, '❓'), state, 'recall', history))
            if index % 3 == 0:
                other = ALPHABET[b]
                correction = record(f'Actually, I prefer {other} instead.', ('🔄', '💖', other),
                                    ('👌', '💖', other), 'correction', history)
                result[name].append(correction)
                updated = pack_history([(first['state'], first['reply']), (correction['state'], correction['reply'])])
                result[name].append(record(question if mood == ('💖',) else 'What do I like now?',
                                          ('🧠', '💖', '❓'), ('💖', other), 'updated_recall', updated))
    return result


def write():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    data = splits()
    for name, rows in data.items():
        (DIRECTORY / f'{name}.jsonl').write_text(
            ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    meanings = {ALPHABET[index]: CATALOG[index]['unicode_name'] for index in range(len(ALPHABET))}
    meanings.update({'👤': 'user turn', '🤖': 'assistant turn', '🔹': 'turn boundary',
                     '💖': 'positive preference', '💔': 'negative preference', '🔄': 'correction or cycle',
                     '🧠': 'recall from conversation', '🎯': 'request for an action plan',
                     '❓': 'question or unresolved detail', '⁉️': 'underspecified request'})
    (DIRECTORY / 'meanings.json').write_text(json.dumps(meanings, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({name: len(rows) for name, rows in data.items()}, indent=2))


if __name__ == '__main__':
    write()
