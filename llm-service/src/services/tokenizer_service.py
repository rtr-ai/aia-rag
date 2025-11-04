import tiktoken

class TokenizerService:
    def __init__(self):
        """
        Initializes the TokenizerService.
        """
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text):
        """
        Counts the number of tokens in a given text.

        Parameters:
        text (str): The input text to tokenize.

        Returns:
        int: The number of tokens in the text.
        """
        tokens = self.encoder.encode(text)
        return len(tokens)