# || IPython
try:
    from IPython.display import Markdown, display
except ModuleNotFoundError:
    Markdown = str
    display = print

# || python 3.10 zip(strict)
from itertools import zip_longest
from typing import Iterable, Iterator, Tuple, TypeVar, overload

_T1 = TypeVar('_T1')
_T2 = TypeVar('_T2')
_T3 = TypeVar('_T3')


@overload
def zip_strict(__iter1: Iterable[_T1]) -> Iterator[Tuple[_T1]]: ...


@overload
def zip_strict(__iter1: Iterable[_T1],
               __iter2: Iterable[_T2]) -> Iterator[Tuple[_T1, _T2]]: ...


@overload
def zip_strict(__iter1: Iterable[_T1],
               __iter2: Iterable[_T2],
               __iter3: Iterable[_T3]) -> Iterator[Tuple[_T1, _T2, _T3]]: ...


def zip_strict(*iterables, strict=True):
    if len(iterables) < 2:
        return zip(*iterables)
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if any(sentinel is c for c in combo):
            raise ValueError('zip() iterables have different lengths')
        yield combo
