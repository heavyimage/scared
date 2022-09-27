import numpy as _np
from ..first_order import _center
from ._base import _combination


def _product(el1, el2):
    return (el1 * el2)


def _difference(el1, el2):
    return (el1 - el2)


def _centered(function, mean):
    return lambda traces: function(_center(traces, mean))


def _absolute(function):
    return lambda traces: _np.absolute(function(traces))


class Difference:
    """Difference combination preprocess for High Order analysis.

    Args:
        frame_1 (slice or iterable, default=...): first traces frame that will be taken.
        frame_2 (slice or iterable, default=None): second optional traces frame that will be taken.
        mode (str, default='full'): Combination mode between `'full'` and `'same'` values.
            In `'same'` mode, each point of `frame_1` will be combined with its corresponding point in `frame_2`.
            The two frames needs to be provided and of the same length when using this mode.
            In `'full'` mode, each point of `frame_1` is combined with full `frame_2` if it is provided,
            otherwise with the frame between the current point position in `frame_1` and the end of the frame if `distance` is None,
            else with a subframe starting at the current point position in `frame_1` and of size equals to `distance`.
        distance (integer, default=None): size of the frame to combine with each point of `frame_1`. This parameter is not available if `frame_2` is provided.

    """

    def __new__(cls, frame_1=..., frame_2=None, mode='full', distance=None):
        return _combination(
            _difference, frame_1=frame_1, frame_2=frame_2, mode=mode, distance=distance
        )


class Product:
    """Product combination preprocess for High Order analysis.

    Args:
        frame_1 (slice or iterable, default=...): first traces frame that will be taken.
        frame_2 (slice or iterable, default=None): second optional traces frame that will be taken.
        mode (str, default='full'): Combination mode between `'full'` and `'same'` values.
            In `'same'` mode, each point of `frame_1` will be combined with its corresponding point in `frame_2`.
            The two frames needs to be provided and of the same length when using this mode.
            In `'full'` mode, each point of `frame_1` is combined with full `frame_2` if it is provided,
            otherwise with the frame between the current point position in `frame_1` and the end of the frame if `distance` is None,
            else with a subframe starting at the current point position in `frame_1` and of size equals to `distance`.
        distance (integer, default=None): size of the frame to combine with each point of `frame_1`. This parameter is not available if `frame_2` is provided.

    """

    def __new__(cls, frame_1=..., frame_2=None, mode='full', distance=None):
        return _combination(
            _product, frame_1=frame_1, frame_2=frame_2, mode=mode, distance=distance
        )


class CenteredProduct(Product):
    """Centered product combination preprocess for High Order analysis.

    Args:
        frame_1 (slice or iterable, default=...): first traces frame that will be taken.
        frame_2 (slice or iterable, default=None): second optional traces frame that will be taken.
        mode (str, default='full'): Combination mode between `'full'` and `'same'` values.
            In `'same'` mode, each point of `frame_1` will be combined with its corresponding point in `frame_2`.
            The two frames needs to be provided and of the same length when using this mode.
            In `'full'` mode, each point of `frame_1` is combined with full `frame_2` if it is provided,
            otherwise with the frame between the current point position in `frame_1` and the end of the frame if `distance` is None,
            else with a subframe starting at the current point position in `frame_1` and of size equals to `distance`.
        distance (integer, default=None): size of the frame to combine with each point of `frame_1`. This parameter is not available if `frame_2` is provided.
        mean (numpy.ndarray, default=None): a mean array with compatible size with traces. If it None, the mean of provided traces is computed.

    """

    def __new__(cls, frame_1=..., frame_2=None, mode='full', distance=None, mean=None):
        return _centered(
            _combination(
                _product, frame_1=frame_1, frame_2=frame_2, mode=mode, distance=distance
            ), mean=mean)


class AbsoluteDifference:
    """Absolute difference combination preprocess for High Order analysis.

    Args:
        frame_1 (slice or iterable, default=...): first traces frame that will be taken.
        frame_2 (slice or iterable, default=None): second optional traces frame that will be taken.
        mode (str, default='full'): Combination mode between `'full'` and `'same'` values.
            In `'same'` mode, each point of `frame_1` will be combined with its corresponding point in `frame_2`.
            The two frames needs to be provided and of the same length when using this mode.
            In `'full'` mode, each point of `frame_1` is combined with full `frame_2` if it is provided,
            otherwise with the frame between the current point position in `frame_1` and the end of the frame if `distance` is None,
            else with a subframe starting at the current point position in `frame_1` and of size equals to `distance`.
        distance (integer, default=None): size of the frame to combine with each point of `frame_1`. This parameter is not available if `frame_2` is provided.

    """

    def __new__(cls, frame_1=..., frame_2=None, mode='full', distance=None):
        return _absolute(
            _combination(_difference, frame_1=frame_1, frame_2=frame_2, mode=mode, distance=distance)
        )
