def input_assert(message, choices) -> str:
    output = input(message).lower()
    if output not in choices:
        print('Wrong input!')
        return input_assert(message, choices)
    else:
        return output
