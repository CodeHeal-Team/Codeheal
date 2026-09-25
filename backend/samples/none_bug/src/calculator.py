def add(a, b):
    return a + b


def divide(a, b):
    if a is None or b is None:
        raise ValueError("None value provided")
    return a / b
