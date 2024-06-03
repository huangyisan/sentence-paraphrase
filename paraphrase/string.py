def cut_sent(content):
	for s in content.split('. '):
		yield s

