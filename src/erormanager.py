
def EROR(erortype, text):
    if erortype == 'IM':
        raise ValueError(f"{text}.")