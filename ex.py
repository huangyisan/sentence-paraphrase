import sys
import os
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

def cut_sent(content):
	for s in content.split('. '):
		yield s

def get_response(input_text,num_return_sequences,num_beams):
  batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  # 返回最长的字符串
  return max(tgt_text, key=len)

def read_from_file(file_name):
	with open(file_name, "r") as f:
		contents = f.readlines()
		return contents

def write_to_file(file_name, content):
	# 每次都会在原有的基础上追加内容
	with open(file_name, "a+") as f:
		f.write(content)
		
#统计文本内容的单词个数
def count_words(res_file_name):
	with open(res_file_name, 'r') as f:
		content = f.read()
	words = content.split()
	return len(words)

if __name__ == "__main__":
  num_beams = 6
  num_return_sequences = 5
    # 获取传入的第一个参数
  if len(sys.argv) <= 1:
    sys.exit("请传入文件名")
  else:
    trans_res_file_name = sys.argv[1]
    file_name = "origin.txt"
    # trans_res_file_name = "Helsinki-NLP_opus-mt-zh-en_trans_res.txt"
    parent_dir = os.path.dirname(os.getcwd())
    file_path = os.path.join(parent_dir, file_name)
    contents = read_from_file(file_path)
    for content in contents:
      for i in  cut_sent(content):
        # print(i)
        write_to_file(trans_res_file_name,get_response(i,num_return_sequences,num_beams)+' ')
      write_to_file(trans_res_file_name, '\n')

  print("重写完成")
  print(f"重写结果保存在{trans_res_file_name}文件中")
  # 内容字数
  print("重写内容字数：", count_words(trans_res_file_name))



