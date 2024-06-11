import re
import torch
from loguru import logger
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

class Pegasus:
    def __init__(self):
        self.model_name = 'tuner007/pegasus_paraphrase'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.debug("Use device: {}".format(self.device))
        try:
            logger.debug("Try to load models from local")
            self.tokenizer = PegasusTokenizer.from_pretrained("./models/")
            self.model = PegasusForConditionalGeneration.from_pretrained("./models/").to(self.device)
            logger.debug("Load models from local")
        except BaseException:
            logger.debug("try to load models from huggingface cache")
            self.tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
            self.model = PegasusForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
            logger.debug("Load models from huggingface cache")
        self.num_return_sequences = 1
        self.num_beams = 1
        self.text = ''
        self.all_response = []
        self.response = []

    def set_text(self, text):
        self.text = text

    def set_num_return_sequences(self, num_return_sequences):
        self.num_return_sequences = num_return_sequences
    def set_num_beams(self, num_beams):
        self.num_beams = num_beams

    def is_beams_less_than_sequences(self):
        if self.num_beams < self.num_return_sequences:
            return True
        return False
    def get_response(self, text):
        self.all_response.append(f"{text}\n↓====================↓")
        batch = self.tokenizer([text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(self.device)
        translated = self.model.generate(**batch,max_length=60,num_beams=self.num_beams, num_return_sequences=self.num_return_sequences, temperature=1.5)
        tgt_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        for i, s in enumerate(tgt_text,1):
            logger.debug(f'translate {i}: {s}')
        self.all_response.extend(tgt_text)
        self.all_response.append('↑====================↑\n')
        self.response.append(max(tgt_text, key=len))

    def paragraph_split(self):
        paragraphs = re.split(r'\n\s*\n', self.text.strip())
        return paragraphs
    def eng_split(self, paragraph):
        sentence_pattern = r'([A-Z][^\.!?]*[\.!?])'
        sentences = re.findall(sentence_pattern, paragraph)
        for s in sentences:
            logger.debug(f'Sentence: {s}')
            yield s

    def make_sentences(self):
        for paragraph in self.paragraph_split():
            for s in self.eng_split(paragraph):
                self.get_response(s)
            self.response.append('\n')
    
    def exec(self, text, num_beams, num_return_sequences):
        self.clean()
        self.set_num_beams(num_beams)
        self.set_num_return_sequences(num_return_sequences)
        if self.is_beams_less_than_sequences():
            return '错误: num_beams 参数必须大于或者等于 num_return_sequences', '错误: num_beams 参数必须大于或者等于 num_return_sequences'
        logger.debug(f'num_beams: {num_beams}, num_return_sequences: {num_return_sequences}')
        self.set_text(text)
        self.make_sentences()
        logger.info(f"result: {self.response}")
        if self.response == ['\n']:
            return '错误: 无法正常分句, 请检查输入是否为英文句子, 查看注意内容第一点', '错误: 无法正常分句, 请检查输入是否为英文句子, 查看注意内容第一点'
        return ' '.join(self.response)
        # return ' '.join(self.response), '\n'.join(self.all_response)

    def clean(self):
        self.text = ''
        self.response = []
        self.all_response = []