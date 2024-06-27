import re

def getPatronFile(baseName, expresionRegular, posicionPatron):
    """
    Extracts a specific pattern from a given base name using a regular expression.

    Args:
        baseName (str): The base name from which to extract the pattern.
        expresionRegular (str): The regular expression pattern to match against the base name.
        posicionPatron (int): The position of the desired pattern group.

    Returns:
        str or None: The extracted pattern if found, None otherwise.
    """
    patron = re.match(rf'{expresionRegular}', baseName)
    if patron:
        return patron.group(posicionPatron)
    else:
        return None