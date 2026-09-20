"""Domain-level errors mapped to HTTP 422 with a precise location."""


class DomainError(Exception):
    """A request-body validation failure detected by the engine.

    ``loc`` mirrors FastAPI's validation-error ``loc`` path (without the
    leading ``"body"`` element, which is added by the exception handler).
    """

    def __init__(self, loc: list, msg: str):
        super().__init__(msg)
        self.loc = list(loc)
        self.msg = msg
