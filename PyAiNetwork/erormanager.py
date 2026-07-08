
def EROR(erortype, text):
    if erortype == 'IM':
        raise ValueError(f"Invalid {text} Please read the documentation")