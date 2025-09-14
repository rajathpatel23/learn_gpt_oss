import tiktoken

def count_tokens(text, enc):
    return len(enc.encode(text))


enc = tiktoken.get_encoding("cl100k_base")

text = "Lowest -> low + est :)"

ids = enc.encode(text)
print(ids)

dec = enc.decode(ids)
print(dec)

tokens = [enc.decode([t]) for t in ids]

for t_id, t_token in zip(ids, tokens):
    print(f"{t_id} -> {t_token}")

print(count_tokens(text, enc))


ids_with_specials = enc.encode("Some text <|endoftext|>", allowed_special="all")

print(ids_with_specials)

def tokenize_with_gpt(s: str, encoding_name: str = "cl100k_base"):
    enc = tiktoken.get_encoding(encoding_name)
    ids = enc.encode(s, allowed_special="all")
    pieces = [enc.decode([i]) for i in ids]
    return ids, pieces

ids, pieces = tokenize_with_gpt("lowest → low + est 😊")
print("IDs:", ids)
print("Pieces:", pieces)


