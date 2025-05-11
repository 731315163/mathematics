import sys
from collections.abc import MutableSequence, Sequence
from datetime import datetime, timedelta
from typing import Any, TypeAlias, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray

DatetimeType: TypeAlias = datetime | np.datetime64 | pd.Timestamp

TimedeltaType: TypeAlias = timedelta | np.timedelta64 | pd.Timedelta

SequenceType: TypeAlias = MutableSequence | Sequence | NDArray


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
    pd.Timestamp,
    pd.Timedelta,
    Any,
)


if sys.version_info < (3, 10):
    SequenceGenericType: TypeAlias = "Union[MutableSequence[T], Sequence[T],NDArray]"
else:
    SequenceGenericType: TypeAlias = MutableSequence[T] | Sequence[T] | np.ndarray
