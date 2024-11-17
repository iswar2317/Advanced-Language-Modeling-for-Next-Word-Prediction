import re

def tokenize(text):
    whitespaces = r'^\s*$'
    url_pattern = r'(https://www\.|http://www\.|https://|http://)?[a-zA-Z0-9]{2,}\.[a-zA-Z]{2,}(\.[a-zA-Z]{2,})?(/[a-zA-Z]{2,})*'
    email_pattern = r'[a-zA-Z0-9\.]+@[a-zA-Z0-9\.\d]+'
    number_pattern = r'\d+(,(\d+))*(\.(\d+))?%?'
    hashtag_pattern = r'#[a-zA-Z\d]+'
    mention_pattern = r'@[a-zA-Z\.\d_]+'
    apos=r'\''


    text=  re.sub(whitespaces,"",text) 
    text = re.sub(email_pattern, "<MAILID>", text)
    text = re.sub(url_pattern, "<URL>", text)
    text = re.sub(number_pattern, "<NUM>", text)
    text = re.sub(hashtag_pattern, "<HASHTAG>", text)
    text = re.sub(mention_pattern, "<MENTION>", text)
    text = re.sub(apos, "", text)



    text = re.sub(r"\b(Dr|Mr|Mrs|Ms)\.", r"\1", text)


    sentences = re.split(r'(?<=[\.\?!])\s+', text)

    tokenized_text = []
    for sentence in sentences:

        words = re.findall(r'<[^>]+>|\b\w+\b|[^\w\s<]', sentence)
        if words:
            tokenized_text.append(words)

    return tokenized_text

def generate_ngrams(tokens, N):
    ngrams = [tuple(tokens[i:i+N]) for i in range(len(tokens)-N+1)]
    return ngrams

temp="Is that what you mean? I am unsure."
output=tokenize(temp)
print(output)

# corpus_path='/Users/iswarmahapatro/sem-2/NLP/assignments/assign1/Pride and Prejudice - Jane Austen.txt'
# with open(corpus_path, 'r', encoding='utf-8') as file:
#     note=file.read()
#     output=tokenize(note)
#     print(output)
    



