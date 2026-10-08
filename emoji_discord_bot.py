"""Run a stateless Discord bot that replies directly with emojis."""

import argparse
import asyncio
import logging
import os

from emoji_local_chat import EmojiLocalChat


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)
MAX_MESSAGE_LENGTH = 4000
ALLOWED_CHANNEL_NAME = '🫪'


def message_text(message):
    if message.author.bot or message.channel.name != ALLOWED_CHANNEL_NAME:
        return None
    text = message.content.strip()
    return text or None


def create_client(chat, discord):
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)
    requests = asyncio.Semaphore(2)

    @client.event
    async def on_ready():
        logger.info('Emoji bot ready as %s', client.user)

    @client.event
    async def on_message(message):
        text = message_text(message)
        if text is None:
            return
        if len(text) > MAX_MESSAGE_LENGTH:
            await message.reply('🙅📏', mention_author=False,
                                allowed_mentions=discord.AllowedMentions.none())
            return

        async with requests:
            async with message.channel.typing():
                try:
                    result = await asyncio.to_thread(chat.answer, text)
                except Exception:
                    logger.exception('Emoji reply failed')
                    await message.reply('🤷', mention_author=False,
                                        allowed_mentions=discord.AllowedMentions.none())
                    return
        await message.reply(result['reply'], mention_author=False,
                            allowed_mentions=discord.AllowedMentions.none())

    return client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    token = os.environ.get('DISCORD_BOT_TOKEN')
    if not token:
        parser.error('Set DISCORD_BOT_TOKEN before starting the bot')
    try:
        import discord
    except ImportError as error:
        raise SystemExit('Install dependencies with: python -m pip install -r requirements.txt') from error

    create_client(EmojiLocalChat(), discord).run(token)


if __name__ == '__main__':
    main()
