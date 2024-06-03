import re
import torch
from loguru import logger
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

class Pegasus:
    def __init__(self, num_return_sequences, num_beams):
        self.model_name = 'tuner007/pegasus_paraphrase'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(self.model_name).to(self.device)

        self.num_return_sequences = num_return_sequences
        self.num_beams = num_beams
        self.text = ''
        self.split_text = []
        self.response = []

    def set_text(self, text):
        self.text = text

    def get_response(self, text):
        batch = self.tokenizer([text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(self.device)
        translated = self.model.generate(**batch,max_length=60,num_beams=self.num_beams, num_return_sequences=self.num_return_sequences, temperature=1.5)
        tgt_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        for i, s in enumerate(tgt_text,1):
            logger.debug(f'translate {i}: {s}')
        self.response.extend(tgt_text)
        logger.debug(f'current self.response is {self.response}')
        # return max(tgt_text, key=len)

    def eng_split(self):
        sentence_pattern = r'([A-Z][^\.!?]*[\.!?])'
        sentences = re.findall(sentence_pattern, self.text)
        for s in sentences:
            logger.debug(f'Sentence:  {s}')
            yield s

    def make_sentences(self):
        for s in self.eng_split():
            self.get_response(s)
        # return self.response
    
    def exec(self, text):
        self.set_text(text)
        self.make_sentences()
        return self.response
    
    def clean(self):
        self.text = ''
        self.split_text = []
        self.response = []