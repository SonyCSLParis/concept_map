import spacy
import time
import openai
from datetime import datetime
import requests
from typing import Union
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from settings import API_KEY_GPT

openai.api_key = API_KEY_GPT

class TextSummarizer:
    def __init__(self, method: str,
                 api_key_gpt: Union[str, None] = None,
                 engine: Union[str, None] = None,
                 temperature: Union[str, None] = None,
                 summary_percentage: Union[str, None] = None,
                 num_sentences: Union[int, None] = None):
        self.method_p = ["lex-rank", "chat-gpt"]
        self.check_params(method=method, api_key_gpt=api_key_gpt, engine=engine,
                          temperature=temperature, summary_percentage=summary_percentage,
                          num_sentences=num_sentences)

        self.method = method
        self.api_key_gpt = api_key_gpt
        self.engine = engine
        self.temperature = temperature
        self.summary_percentage = summary_percentage
        self.num_sentences = num_sentences

        self.nlp = spacy.load("en_core_web_lg")
        self.limit = 16385
    
    def check_params(self, method, api_key_gpt, engine, temperature, summary_percentage, num_sentences):
        if method not in self.method_p:
            raise ValueError(f"Invalid summary method: {method}, possible options are: {self.method_p}")
        if method == "chat-gpt":
            if (not isinstance(api_key_gpt, str)) or (not api_key_gpt):
                raise ValueError(f"For {method} summarisation, `api_key_gpt` must be non-empty string")
            if (not isinstance(engine, str)) or (not engine):
                raise ValueError(f"For {method} summarisation, `engine` must be non-empty string")
            if not isinstance(summary_percentage, int):
                raise ValueError(f"For {method} summarisation, `summary_percentage` must be int")
            if not isinstance(temperature, float):
                raise ValueError(f"For {method} summarisation, `temperature` must be int")
        if method == "lex-rank":
            if not isinstance(num_sentences, int):
                raise ValueError(f"For {method} summarisation, `num_sentences` must be int")

    def calculate_max_tokens(self, text: str, percentage: int):
        doc = self.nlp(text)
        total_tokens = len(doc)
        max_tokens = int(total_tokens * (percentage / 100))

        len_prompt = 500
        if total_tokens + max_tokens < self.limit - len_prompt:
            return max_tokens, text
        # Else: only considering text until a certain number of tokens
        # st. the percentage is respected
        new = int((self.limit - len_prompt)/(1+percentage/100))
        text = doc[:new].text
        max_tokens = int(new * (percentage / 100))
        return max_tokens, text

    def generate_summary_with_gpt(self, text: str, summary_percentage: int, temperature: float):
        max_tokens, text = self.calculate_max_tokens(text, summary_percentage)

        # endpoint = f"https://api.openai.com/v1/engines/{self.engine}/completions"
        # response = requests.post(
        #     endpoint,
        #     headers={
        #         "Authorization": f"Bearer {self.api_key_gpt}",
        #         "Content-Type": "application/json",
        #     },
        #     json={
        #         "prompt": f'Please summarize the following text:\n"{text}"\n\nSummary:',
        #         "max_tokens": max_tokens,
        #         "temperature": temperature,
        #     }
        # )

        # if response.status_code == 429:
        #     # Handle rate-limiting by waiting for a specified duration
        #     wait_time = int(response.headers.get("Retry-After", 10))
        #     print(f"Rate limited. Waiting for {wait_time} seconds.")
        #     time.sleep(wait_time)
        #     return self.generate_summary_with_gpt(text)  # Retry the request

        # response.raise_for_status()
        # result = response.json()

        summary = openai.Completion.create(
            engine=self.engine,
            prompt=f"Please summarize this text in {max_tokens} words or fewer:\n{text}\n\nSummary:",
            max_tokens=max_tokens,
            temperature=self.temperature,
        )

        try:
            summary = summary["choices"][0]["text"].strip()
            summary = summary.split("\n")[0]
            return summary
        except Exception as e:
            print(e)
            raise ValueError("Something went wrong with the summary")

    def generate_lex_rank_summary(self, text: str, num_sentences: int):
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary = summarizer(parser.document, num_sentences)
        return " ".join(str(sentence) for sentence in summary)
    
    def __call__(self, text: str):
        """
        - text: input text to summarise
        - output = generated summary
        """
        if self.method == "lex-rank":
            return self.generate_lex_rank_summary(text, self.num_sentences)
        # self.method == "chat-gpt"
        return self.generate_summary_with_gpt(text, self.summary_percentage, self.temperature)


if __name__ == '__main__':
    summarizer = TextSummarizer(
        api_key_gpt=API_KEY_GPT,
        engine="davinci-002",
        method="chat-gpt",
        summary_percentage=60,
        temperature=0.0)

    TEXT = """
        James Madison’s Biography
James & Dolley Madison Born on March 16, 1751 at his grandmother’s home in Port Conway, Virginia, James Madison was the eldest of the twelve children of James Madison Sr. and Nelly Conway Madison.
His early years were spent at Mount Pleasant, the first house built on the Montpelier plantation.
At age 12, Madison’s father sent him to Donald Robertson’s school in King and Queen County.
There Madison studied arithmetic and geography, learned Latin and Greek, acquired a reading knowledge of French, and began to study algebra and geometry.
Madison never forgot his teacher, later acknowledging “all that I have been in life I owe largely to that man.”
After further study with a private tutor at Montpelier, Madison enrolled in college at the College of New Jersey (today known as Princeton University), earning a bachelor’s degree in 1771.
He continued his education at Princeton through the next winter, studying Hebrew and ethics.
Madison overworked himself in order to complete two years of coursework in one.
In poor health, he returned to Montpelier, where he continued to read on a variety of topics, particularly law.
        """
    
    summary = summarizer(TEXT)
    print(f"ChatGPT summary: \n {summary}")

    summarizer = TextSummarizer(
        method="lex-rank", num_sentences=3)
    summary = summarizer(TEXT)
    print(f"LexRank summary: \n {summary}")
