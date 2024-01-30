import time

import requests
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from settings import *

class TextSummarizer:
    def __init__(self, api_key_gpt, engine):
        self.api_key_gpt = api_key_gpt
        self.engine = engine


    def calculate_max_tokens(self, text, percentage):
        doc = nlp(text)
        total_tokens = len(doc)
        max_tokens = int(total_tokens * (percentage / 100))
        return max_tokens

    def generate_summary_with_gpt(self, text, summary_percentage, temperature):
        max_tokens = self.calculate_max_tokens(text, summary_percentage)

        endpoint = f"https://api.openai.com/v1/engines/{self.engine}/completions"
        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key_gpt}",
                "Content-Type": "application/json",
            },
            json={
                "prompt": text,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )

        if response.status_code == 429:
            # Handle rate-limiting by waiting for a specified duration
            wait_time = int(response.headers.get("Retry-After", 10))
            print(f"Rate limited. Waiting for {wait_time} seconds.")
            time.sleep(wait_time)
            return self.generate_summary_with_gpt(text)  # Retry the request

        response.raise_for_status()
        result = response.json()
        summary = result["choices"][0]["text"].strip()
        return summary

    def generate_lex_rank_summary(self, text, num_sentences=3):
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary = summarizer(parser.document, num_sentences)
        return " ".join(str(sentence) for sentence in summary)


if __name__ == '__main__':
    API_KEY_GPT = "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"
    summarizer = TextSummarizer(api_key_gpt=API_KEY_GPT, engine="davinci-002")

    TEXT = """
        The 52-story, 1.7-million-square-foot 7 World Trade Center is a benchmark of innovative design, safety, and sustainability.
        7 WTC has drawn a diverse roster of tenants, including Moody's Corporation, New York Academy of Sciences, Mansueto Ventures, MSCI, and Wilmer
        Hale.
        """

    lex_rank_summary = summarizer.generate_lex_rank_summary(TEXT)
    gpt_summary = summarizer.generate_summary_with_gpt(TEXT, summary_percentage=80, temperature=0.7)

    print("LexRank Summary:")
    print(lex_rank_summary)

    print("\nGPT Summary:")
    print(gpt_summary)
