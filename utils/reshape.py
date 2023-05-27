
def __list_to_string(list_of_strings):
    return ' '.join([str(x) for x in list_of_strings])


def get_reshape_string(qtype, ktype):
    qtype = __list_to_string(qtype)
    ktype = __list_to_string(ktype)
    return f'({qtype}) ({ktype})'
