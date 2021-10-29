from typing import List


def input_assert(message: str, choices: List[str]) -> str:
    """
    Adds functionality to the python function `input` to limit the choices that can be returned

    Args:
        message: message to user
        choices: list containing possible choices that can be returned

    Returns:
        input returned by user
    """
    output = input(message).lower()
    if output not in choices:
        print(f"Wrong input! Your input should be one of the following: {', '.join(choices)}")
        return input_assert(message, choices)
    else:
        return output
