from typing import Any, Iterable, TypeVar, Sequence

from utils import print_error

T = TypeVar('T')


def try_open_read(filename: str,
                  filework,
                  *args,
                  **kwargs) -> Any:
    try:
        with open(filename, "r") as file:
            return filework(file, *args, **kwargs)
    except FileNotFoundError:
        print_error(f"File not found: {filename}")
        raise
    except IOError as error:
        print_error(error)
        raise


def list_of(items: Iterable[T]) -> list[T]:
    return items if items is list else list(items)


def modify_tuple(old_tuple: tuple, index: int, new_value: Any) -> tuple:
    return old_tuple[0:index] + (new_value,) + old_tuple[index + 1:]


def chunk(sequence: Sequence[T],
          chunk_size: int) -> Iterable[Iterable[T]]:
    for i in range(0, len(sequence), chunk_size):
        yield sequence[i:(i + chunk_size)]


def str_or_default(x: Any):
    return str(x) if x is not None else ""
