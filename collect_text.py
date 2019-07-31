import pandas as pd
import os, sys, string
from utils import save_text

DATA_DIR = 'training_data'

datas = []

for file in os.listdir(DATA_DIR):
    print(file)
    datas.append(pd.read_csv(DATA_DIR+'/'+file))

data = pd.concat(datas)

comments = data['commentText']

print('Total Comments: ',len(comments))

clean_comments = []

for comment in comments:
    check_comment = str(comment).lower().split()
    if 'robot' in check_comment or 'build' in check_comment or 'make' in check_comment:
        clean_comments.append(str(str(comment).encode('utf-8'))[2:-1])

print('Total Robot Comments: ',len(clean_comments))

clean_comments = set(clean_comments)

print('Unique Robot Comments: ',len(clean_comments))

cleaned = []

for comment in clean_comments:
    tokens = comment.split()
    table = str.maketrans('','',string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    cleaned.append(' '.join(tokens))

save_text(cleaned, DATA_DIR+'/comments.txt')
