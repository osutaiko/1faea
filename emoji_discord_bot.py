"""Run an emoji Discord bot with bounded per-user conversation memory."""

import argparse
import asyncio
import logging
import os

from emoji_local_chat import EmojiLocalChat, remember_turn


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
    requests = asyncio.Semaphore(1)
    histories = {}

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

        session = (message.guild.id, message.channel.id, message.author.id)
        async with requests:
            async with message.channel.typing():
                try:
                    history = histories.setdefault(session, [])
                    reply = await asyncio.to_thread(chat.answer, text, history.copy())
                except Exception:
                    logger.exception('Emoji reply failed')
                    await message.reply('🤷', mention_author=False,
                                        allowed_mentions=discord.AllowedMentions.none())
                    return
                remember_turn(history, text, reply)
        await message.reply(reply, mention_author=False,
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

    chat = EmojiLocalChat()
    try:
        logger.info('Checking the local model server; first startup may download model weights')
        chat.start_server()
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    try:
        create_client(chat, discord).run(token)
    finally:
        chat.close_server()


if __name__ == '__main__':
    main()
