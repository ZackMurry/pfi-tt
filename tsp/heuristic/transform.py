from gymnasium import spaces
import numpy as np
import itertools
import numbers
from collections import namedtuple

Transform = namedtuple('Transform', ['original', 'target', 'convert_to', 'convert_from'])
def assert_space(space):
    """ Raise a `TypeError` exception if `space` is not a `gym.spaces.Space`. """
    if not isinstance(space, gym.Space):
        raise TypeError("Expected a gym.spaces.Space, got {}".format(type(space)))


def is_discrete(space):
    """ Checks if a space is discrete. A space is considered to
        be discrete if it is derived from Discrete, MultiDiscrete
        or MultiBinary.
        A Tuple space is discrete if it contains only discrete 
        subspaces.
        :raises TypeError: If the space is no `gym.Space`.
    """
    assert_space(space)

    if isinstance(space, (spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary)):
        return True
    elif isinstance(space, spaces.Box):
        return False
    elif isinstance(space, spaces.Tuple):
        return all(map(is_discrete, space.spaces))

    raise NotImplementedError("Unknown space {} of type {} supplied".format(space, type(space)))


def is_compound(space):
    """ Checks whether a space is a compound space. These are non-scalar
        `Box` spaces, `MultiDiscrete`, `MultiBinary` and `Tuple` spaces
        (A Tuple space with a single, non-compound subspace is still considered
        compound).
        :raises TypeError: If the space is no `gym.Space`.
    """
    assert_space(space)

    if isinstance(space, spaces.Discrete):
        return False
    elif isinstance(space, spaces.Box):
        return len(space.shape) != 1 or space.shape[0] != 1
    elif isinstance(space, (spaces.MultiDiscrete, spaces.MultiBinary)):
        return True
    elif isinstance(space, spaces.Tuple):
        return True

    raise NotImplementedError("Unknown space {} of type {} supplied".format(space, type(space)))


def is_flat(space):
    """
    Checks whether space is a flat space. Flat spaces ore either Discrete, or Box spaces with rank less or equal one.
    :param gym.Space space: The space to check for flatness.
    :return: Whether the space is flat.
    """
    assert_space(space)

    if isinstance(space, spaces.Discrete):
        return True
    elif isinstance(space, spaces.Box):
        return len(space.shape) <= 1
    elif isinstance(space, (spaces.MultiDiscrete, spaces.MultiBinary)):
        return False
    elif isinstance(space, spaces.Tuple):
        return False

    raise NotImplementedError("Unknown space {} of type {} supplied".format(space, type(space)))


# Flattening
def flatten(space):
    """
    Flattens a space, which means that for continuous spaces (Box)
    the space is reshaped to be of rank 1, and for multidimensional
    discrete spaces a single discrete action with an increased number
    of possible values is created.
    Please be aware that the latter can be potentially pathological in case
    the input space has many discrete actions, as the number of single discrete
    actions increases exponentially ("curse of dimensionality").
    :param gym.Space space: The space that will be flattened
    :return Transform: A transform object describing the transformation
            to the flattened space.
    :raises TypeError, if `space` is not a `gym.Space`.
            NotImplementedError, if the supplied space is neither `Box` nor
            `MultiDiscrete` or `MultiBinary`, and not recognized as
            an already flat space by `is_compound`.
    """
    # no need to do anything if already flat
    if is_flat(space):
        return Transform(space, space, _identity, _identity)

    if isinstance(space, spaces.Box):
        shape = space.low.shape
        lo = space.low.flatten()
        hi = space.high.flatten()

        def convert(x):
            return np.reshape(x, shape)

        def back(x):
            return np.reshape(x, lo.shape)

        flat_space = spaces.Box(low=lo, high=hi, dtype=space.dtype)
        return Transform(original=space, target=flat_space, convert_from=convert, convert_to=back)

    elif isinstance(space, (spaces.MultiDiscrete, spaces.MultiBinary)):
        if isinstance(space, spaces.MultiDiscrete):
            ranges = [range(0, k, 1) for k in space.nvec]
        elif isinstance(space, spaces.MultiBinary):  # pragma: no branch
            ranges = [range(0, 2) for i in range(space.n)]
        prod   = itertools.product(*ranges)
        lookup = list(prod)
        inverse_lookup = {value: key for (key, value) in enumerate(lookup)}
        flat_space = spaces.Discrete(len(lookup))
        return Transform(original=space, target=flat_space,
                         convert_from=_Lookup(lookup), convert_to=_Lookup(inverse_lookup))

    elif isinstance(space, spaces.Tuple):
        # first ensure all subspaces are flat.
        flat_subs = [flatten(sub) for sub in space.spaces]
        lo = np.concatenate([f.target.low for f in flat_subs])
        hi = np.concatenate([f.target.high for f in flat_subs])
        return Transform(space, target=spaces.Box(low=lo, high=hi), convert_to=_FlattenTuple(flat_subs),
                         convert_from=_DecomposeTuple(flat_subs))

    raise NotImplementedError("Does not know how to flatten {}".format(type(space)))  # pragma: no cover
