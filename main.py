# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "nltk",
#     "prompt-toolkit",
# ]
# ///

import sys
import re
import gzip
from dataclasses import dataclass
from typing import Generator, Tuple, Iterator
from pathlib import Path

from nltk.tokenize import word_tokenize
from prompt_toolkit import prompt
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document


@dataclass
class CompletionResult:
    word: str
    compressed_size: int


def train_gzip_generator(
    training_file_path: Path | str,
) -> Generator[Tuple[str, int], str, None]:
    try:
        with open(training_file_path, "r", encoding="utf-8") as file:
            training_data = file.read()
    except (FileNotFoundError, IOError) as e:
        raise IOError(f"Error reading training file: {e}")

    training_data = re.sub(r"\([^)]*\)", "", training_data)

    tokenized_training_data = word_tokenize(training_data)

    vocabulary = set(tokenized_training_data)

    def generate_next_word(prompt: str) -> Iterator[Tuple[str, int]]:
        tokenized_prompt = word_tokenize(prompt)

        for word in vocabulary:
            text = " ".join(tokenized_training_data + tokenized_prompt + [word])
            compressed_size = len(gzip.compress(text.encode()))
            yield word, compressed_size

    return generate_next_word


class GzipCompleter(Completer):
    def __init__(self, generator: Generator[Tuple[str, int], str, None]) -> None:
        self.generator = generator

    def get_completions(
        self, document: Document, complete_event: bool
    ) -> Iterator[Completion]:
        word_before_cursor = document.text_before_cursor.strip()
        if not word_before_cursor:
            return

        try:
            results = sorted(
                list(self.generator(word_before_cursor)), key=lambda x: x[1]
            )
            for word, _ in results[:1]:
                yield Completion(word, start_position=-len(word_before_cursor))
        except Exception as e:
            print(f"Error generating completions: {e}")


if __name__ == "__main__":
    training_file_path = sys.argv[1] if len(sys.argv) > 1 else None

    if not training_file_path:
        print("Please provide a training file path(eg: python main.py training.txt)")
        raise SystemExit

    generator = train_gzip_generator(training_file_path)
    gzip_completer = GzipCompleter(generator)

    print("Enter a sentence to predict the next word ->")
    while True:
        user_input = prompt(
            "🐍: ", completer=gzip_completer, complete_while_typing=True
        )

        for i, (word, size) in enumerate(
            sorted(generator(user_input), key=lambda item: item[1])[:5], 1
        ):
            print(f"{i}. {word}, Compressed Size: {size}")
        print()

