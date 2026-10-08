"""Run interactive emoji-only chat in the terminal using the local model."""

from emoji_local_chat import EmojiLocalChat, remember_turn


def main():
    chat = EmojiLocalChat()
    history = []
    try:
        chat.start_server()
        print('Emoji chat ready. Type /quit to exit.')
        while True:
            try:
                question = input('> ').strip()
            except EOFError:
                break
            if question == '/quit':
                break
            if not question:
                continue
            try:
                reply = chat.answer(question, history.copy())
            except Exception as error:
                print(f'🤷 Model request failed: {error}', flush=True)
                continue
            print(reply, flush=True)
            remember_turn(history, question, reply)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    finally:
        chat.close_server()


if __name__ == '__main__':
    main()
