import sys
from typing import Iterable, TypeVar, Collection

T = TypeVar('T')


def print_error(*args, **kwargs):
    """
    Prints a message to stderr.
    """
    print(*args, file=sys.stderr, **kwargs)


def index_groups(groups: Iterable[Collection[T]], keys: Collection[T]) -> dict[T, Collection[T]]:
    """
    Indexes a collection of groups by a set of keys.
    :param groups: An iterable of collections to be indexed.
    :param keys: A collection of keys to index the groups by.
    :return: A dictionary where each key is a key from the input collection and each value is the first group that contains that key.
    :raises KeyError: If a key is not found in any of the groups.
    """
    index: dict[T, Collection[T]] = dict()
    for group in groups:
        for key in keys:
            if key in group:
                index[key] = group
                break
        else:
            raise KeyError("Group does not contain a key")
    return index
