import sys
from typing import Any, Iterable, TypeVar, Generator, Sequence

T = TypeVar('T')


def print_error(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def try_open(filename: str,
             filework,
             *args,
             **kwargs) -> Any:
    try:
        with open(filename, "r") as file:
            return filework(file, *args, **kwargs)
    except FileNotFoundError:
        print_error(f"File not found: {filename}")
    except IOError as error:
        print_error(error)


def list_of(items: Iterable[T]) -> list[T]:
    return items if items is list else list(items)


def modify_tuple(old_tuple: tuple, index: int, new_value: Any) -> tuple:
    return old_tuple[0:index] + (new_value,) + old_tuple[index + 1:]


def chunk(sequence: Sequence[T],
          chunk_size: int) -> Iterable[Iterable[T]]:
    for i in range(0, len(sequence), chunk_size):
        yield sequence[i:(i + chunk_size)]
