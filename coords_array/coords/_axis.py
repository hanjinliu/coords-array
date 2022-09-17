from __future__ import annotations
import warnings
from copy import copy
from typing import Any, Hashable, SupportsIndex, Union, Iterable, TYPE_CHECKING
import numpy as np

from ._misc import CoordinateWarning
from ._index import as_index, Index, ScaledIndex
from ..typing import IndexLike

if TYPE_CHECKING:
    from typing_extensions import Self

_Slicable = Union[SupportsIndex, slice, list[int], np.ndarray]
_Real = Union[int, float]

class _AxisBase:
    def __init__(self, name: str):
        self._name = str(name)
    
    def __str__(self) -> str:
        """String representation of the axis."""
        return self._name
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self._name!r}]"
    
    def __eq__(self, other) -> bool:
        """Check equality as a string."""
        return str(self) == other
    
    def __neg__(self) -> Self:
        """Invert axis."""
        return LinearTransformedAxis.from_linear_combination([(-1., self)])
    
    def __add__(self, other: _AxisBase) -> LinearTransformedAxis:
        """Add as a string ans returns a string."""
        if isinstance(other, _AxisBase):
            return LinearTransformedAxis.from_linear_combination([(1., self), (1., other)])
        else:
            raise TypeError(f"Cannot add {type(self)} and {type(other)}.")
    
    def __radd__(self, other: _AxisBase) -> LinearTransformedAxis:
        """Add as a string ans returns a string."""
        if isinstance(other, _AxisBase):
            return LinearTransformedAxis.from_linear_combination([(1., other), (1., self)])
        else:
            raise TypeError(f"Cannot add {type(other)} and {type(self)}.")

    def __sub__(self, other: _AxisBase) -> LinearTransformedAxis:
        """Subtract another axis."""
        return LinearTransformedAxis.from_linear_combination([(1., self), (-1., other)])

    def __mul__(self, coef: _Real) -> LinearTransformedAxis:
        """Multiply by a scalar."""
        return LinearTransformedAxis.from_linear_combination([(coef, self)])
    
    def __rmul__(self, coef: _Real) -> LinearTransformedAxis:
        """Multiply by a scalar."""
        return LinearTransformedAxis.from_linear_combination([(coef, self)])        
    
    @property
    def name(self) -> str:
        """Name of the axis."""
        return self._name
    
    @property
    def scale(self) -> float:
        """Scale of the axis."""
        return 1.0
    
    @property
    def unit(self) -> str | None:
        """Unit of the axis."""
        return None
    

class Axis(_AxisBase):
    """
    An axis object.
    
    This object behaves like a length-1 string as much as possible.
    
    Parameters 
    ----------
    name : str
        Name of axis.
    """
    
    def __init__(
        self,
        name: str,
        *,
        index: Index | None = None,
    ):
        super().__init__(name)
        self._index = index
    
    @classmethod
    def from_size(cls: type[Axis], name: str, size: int) -> Self:
        index = ScaledIndex.arange(size)
        return cls(name, index=index)
    
    def __copy__(self) -> Self:
        """Return a copy of the axis."""
        return self.__class__(self._name, index=self.index)
    
    def __hash__(self) -> int:
        """Hash as a string."""
        return hash(str(self))

    def __repr__(self) -> str:
        _cls = type(self).__name__
        return f"{_cls}(name={self.name!r}, index={self.index!r})"

    @property
    def scale(self) -> float:
        """Physical scale of axis."""
        return self.index.get_scale()
    
    @scale.setter
    def scale(self, value: _Real) -> None:
        """Set physical scale to the axis."""
        value = float(value)
        if value <= 0:
            raise ValueError(f"Cannot set negative scale: {value!r}.")
        self.index = self.index.rescaled(value)
    
    @property
    def unit(self) -> str:
        """Physical scale unit of axis."""
        return self.index.get_unit()
    
    @unit.setter
    def unit(self, value: str | None):
        """Set physical unit to the axis."""
        return self.index.set_unit(value)

    @property
    def index(self) -> Index:
        """Axis index."""
        return self._index
    
    @index.setter
    def index(self, value: IndexLike) -> None:
        """Set axis index."""
        index = as_index(value, self.size)
        self._set_index(index)
    
    def _set_index(self, index: Index) -> None:
        """Set axis index without validation."""
        self._index = index
    
    @property
    def size(self) -> int:
        return self.index.get_size()

    def isin(self, values: IndexLike) -> np.ndarray:
        """Check if labels are in values."""
        if self.index is None:
            raise ValueError("Axis has no coordinates.")
        return np.array([label in values for label in self.index])

    def slice_axis(self, sl: _Slicable) -> Self:
        """
        Return sliced axis.
        
        Parameters
        ----------
        sl : slicable object
            Slice object to apply to the axis.
        """
        if not isinstance(sl, (slice, list)):
            return self

        # slice coordinates
        if isinstance(sl, slice):
            new_index = self.index[sl]
        else:
            new_index = self.index.subset(sl)
        
        return self.__class__(self._name, index=new_index)

    

AxisLike = Union[str, _AxisBase]


class UndefAxis(Axis):
    """Undefined axis object."""
    
    def __init__(self, *args, **kwargs):
        super().__init__("#")
    
    def __repr__(self) -> str:
        return "#undef"
    
    def __hash__(self) -> str:
        return id(self)
    
    def __eq__(self, other) -> bool:
        return isinstance(other, str) and other == "#"

def as_axis(obj: Any, size: int) -> Axis:
    """Convert an object into an ``Axis`` object."""
    if isinstance(obj, str):
        if obj == "#":
            axis = UndefAxis()
        else:
            axis = Axis.from_size(obj, size)
    elif isinstance(obj, Axis):
        axis = copy(obj)
    else:
        raise TypeError(f"Cannot use {type(obj)} as an axis.")
    return axis


class LinearTransformedAxis(_AxisBase):
    """
    Axis that is a linear combination of other axes.
    
    Examples
    --------
    >>> x = Axis("x")
    >>> y = Axis("y")
    >>> u = 0.3 * x + 0.4 * y
    >>> u
    LinearTransformedAxis['0.3x+0.4y']
    >>> u - x
    LinearTransformedAxis['-0.7x+0.4y']
    """

    def __init__(self, name: str, components: dict[_AxisBase, float]):    
        super().__init__(name)
        self._components = components
        self._scale = np.linalg.norm(
            [axis.scale * abs(coef) for axis, coef in components.items()]
        )
        base_units = set(axis.unit for axis in components.keys())
        if len(base_units) == 1:
            self._unit = base_units.pop()
        else:
            warnings.warn("Inconsistent units in transformed axis.", CoordinateWarning)
            self._unit = None
    
    def __hash__(self) -> int:
        """Hash as a string."""
        return hash(tuple(self.components.items()))
    
    @classmethod
    def from_linear_combination(
        cls: type[LinearTransformedAxis],
        components: Iterable[tuple[float, _AxisBase]],
        name: str | None = None,
    ) -> Self:
        """
        Construct a LinearTransformedAxis from a linear combination of other axes.

        Parameters
        ----------
        components : Iterable of (float, Axis)
            Coefficient and axis.
        name : str, optional
            Name of the axis. By default, a simplified expression of the linear combination
            will be used.

        Returns
        -------
        LinearTransformedAxis
            Axis with the given linear combination of other axes.        
        """
        axis_to_coef: dict[_AxisBase, float] = {}
        for k, axis in components:
            if isinstance(axis, UndefAxis):
                raise TypeError("Cannot use undefined axis in a linear combination.")
            else:
                _increment_axis_component(axis_to_coef, k, axis)
            
        if name is None:
            name = "".join(f"{k:+.2g}{axis}" for axis, k in axis_to_coef.items()).lstrip("+")

        return cls(name, components=axis_to_coef)

    @property
    def components(self) -> dict[_AxisBase, float]:
        return self._components
    
    @property
    def scale(self) -> float:
        return self._scale
    
    @property
    def unit(self) -> Any | None:
        return self._unit
    
    @property
    def vector(self) -> np.ndarray:
        """Vector representation of the axis."""
        return np.array(list(self.components.values()), dtype=np.float32)
    
    @property
    def bases(self) -> list[Axis]:
        """Base axes."""
        return list(self.components.keys())
    
    def transform(self, matrix: np.ndarray) -> Self:
        """Transform axis by a matrix."""
        new = self.vector.dot(matrix)
        return self.from_linear_combination(zip(new, self.components.keys()))
    
    def __matmul__(self, matrix: np.ndarray) -> Self:
        """Transform axis by a matrix."""
        return self.transform(matrix)


def _increment_axis_component(dict_: dict, coef: float, axis: Axis) -> None:
    """Increment a component of the axis."""
    if isinstance(axis, LinearTransformedAxis):
        for _axis, _coef in axis.components.items():
            _increment_axis_component(dict_, _coef * coef, _axis)
    else:
        if axis in dict_:
            dict_[axis] += coef
        else:
            dict_[axis] = coef
    return None
