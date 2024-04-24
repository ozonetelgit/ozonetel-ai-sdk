class AuthenticationError(Exception):
    """Authentication Failed."""

    def __init__(self, message="Authentication failed. Please check your credentials."):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.__class__.__name__}: {self.message}'
    
class LimitError(Exception):
    """Limit Quota Exceeded."""

    def __init__(self, message="Limit Exceeded."):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.__class__.__name__}: {self.message}'