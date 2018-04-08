import numpy as np


def check_type(obj, type_of_type):
    # Set 'input_type'
    input_type = None
    if type(obj) in [type,np.dtype]:
        input_type = obj
    else: input_type = type(obj)

    # Check 'type_of_type'
    assert type(type_of_type) is str
    type_of_type_list = list(np.typecodes.keys())
    type_of_type_list_lowercase = [typecode_category.lower() for typecode_category in type_of_type_list]
    assert type_of_type.lower() in type_of_type_list_lowercase
    type_of_type_index = type_of_type_list_lowercase.index(type_of_type.lower())
    type_of_type = type_of_type_list[type_of_type_index]
    
    ## Check if there's matched type in numpy types
    ## .. note that all build-in types are included in numpy types
    type_matched = False
    for typecode in np.typecodes[type_of_type]:
        typ = np.typeDict[typecode]
        if np.dtype(input_type) == typ:
            type_matched = True

    # Return whether the input ndarray's type has a matching type
    return type_matched

def is_real_number(obj):
    is_float = check_type(obj, 'float')
    is_integer = check_type(obj, 'integer')
    return is_float or is_integer

def is_integer(obj):
    return check_type(obj, 'integer')

def is_float(obj):
    return check_type(obj, 'float')

def check_type_ndarray(obj, type_of_type):
    # Check 'obj'
    assert type(obj) is np.ndarray
    assert hasattr(obj, 'dtype')
    return check_type(obj.dtype, type_of_type)

def is_integer_valued_real(obj):
    is_integer_valued = False
    if is_real_number(obj):
        if int(obj) == obj:
            is_integer_valued = True
    return is_integer_valued
