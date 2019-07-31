def load_text(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def clean_text(doc):
    import string, os

    doc = doc.replace('\n',' ')
    tokens = doc.split()
    table = str.maketrans('','',string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    return tokens

def save_text(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()
