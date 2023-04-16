def round_off(number: float) -> float:
    """Round number to the lowest integer close to the number

    Args:
        number (float): _description_

    Returns:
        float: _description_
    """
    number = number - 0.499999999999
    return round(number) 

def cap_off(number: float) -> int:
    """Cap zones starting coordinates to -100 at minimum and 40 maximum

    Args:
        number (float): Starting coordinates

    Returns:
        int: Capped starting coordinates
    """
    if number > 40:
        return 40
    elif number < -40:
        return -100
    else: 
        return number