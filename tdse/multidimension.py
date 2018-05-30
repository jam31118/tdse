import numpy as np
def cyclic_swapaxes(arr):
    """
    # Example
    For natural numbers N1, N2, N4,
    an array with shape (N2,N4,N1) yield a view of the array
    with shape (N1,N2,N4), which is just a result of cyclic translation of axes
    """
    assert isinstance(arr, np.ndarray) 
    for attribute_Name in ['swapaxes','ndim']: assert hasattr(arr, attribute_Name)
    if arr.ndim < 2: raise TypeError("numpy.ndarray with dimension less than 2 cannot perform swapping axes")
    for axes_index in range(arr.ndim-1):
        arr = arr.swapaxes(-(axes_index+1), -(axes_index+2))
    return arr


def cyclically_shift_elements_in_C_order(l):
    """
    # Example
    (34, 21, 59) -> (59, 34, 21)
    """
    new_l = tuple(l[(index-1)%len(l)] for index in range(len(l)))
    return new_l
