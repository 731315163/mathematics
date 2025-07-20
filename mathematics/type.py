import sys
from collections.abc import MutableSequence, Sequence
from datetime import datetime, timedelta
from typing import Any, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray

DatetimeType: TypeAlias = datetime | np.datetime64 

TimedeltaType: TypeAlias = timedelta | np.timedelta64 

SequenceType: TypeAlias = MutableSequence | Sequence | np.ndarray


T = TypeVar(
    "T",
    str,
    int,
    float,
    np.number,
    datetime,
    timedelta,
    np.datetime64,
    np.timedelta64,
   
    Any,
)


if sys.version_info < (3, 10):
    from typing_extensions import Union
    SequenceGenericType: TypeAlias = "Union[MutableSequence[T], Sequence[T],NDArray]"
else:
    SequenceGenericType: TypeAlias = MutableSequence[T] | Sequence[T] | np.ndarray
