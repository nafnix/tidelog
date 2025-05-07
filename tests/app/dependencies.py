from fastapi.exceptions import HTTPException


def bad():
    raise HTTPException(400)


class A:
    def __init__(self, name: int) -> None:
        self.name = name
