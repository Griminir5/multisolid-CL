"""Lazy DAETools symbols used by kinetics expression functions."""

from __future__ import annotations


def _dae_function(name: str, *args):
    from daetools import pyDAE

    return getattr(pyDAE, name)(*args)


def Constant(value):
    return _dae_function("Constant", value)


def Exp(value):
    return _dae_function("Exp", value)


def Log(value):
    return _dae_function("Log", value)


def Max(left, right):
    return _dae_function("Max", left, right)


def Min(left, right):
    return _dae_function("Min", left, right)


def Abs(value):
    return _dae_function("Abs", value)


def Sqrt(value):
    return _dae_function("Sqrt", value)


class _LazyUnit:
    def __init__(self, name: str):
        self.name = name

    def _value(self):
        from daetools import pyDAE  # noqa: F401 - initializes bundled pyUnits path
        import pyUnits

        return getattr(pyUnits, self.name)

    @staticmethod
    def _unwrap(value):
        return value._value() if isinstance(value, _LazyUnit) else value

    def __mul__(self, other):
        return self._value() * self._unwrap(other)

    def __rmul__(self, other):
        return self._unwrap(other) * self._value()

    def __truediv__(self, other):
        return self._value() / self._unwrap(other)

    def __rtruediv__(self, other):
        return self._unwrap(other) / self._value()

    def __pow__(self, power):
        return self._value() ** power


K = _LazyUnit("K")
Pa = _LazyUnit("Pa")
m = _LazyUnit("m")
mol = _LazyUnit("mol")
s = _LazyUnit("s")


__all__ = ("Abs", "Constant", "Exp", "K", "Log", "Max", "Min", "Pa", "Sqrt", "m", "mol", "s")
