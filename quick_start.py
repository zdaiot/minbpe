from minbpe import BasicTokenizer
tokenizer = BasicTokenizer()
text = "aaabdaaabac"
tokenizer.train(text, 256 + 3, verbose=True) # 256 are the byte tokens, then do 3 merges
# text = "ababa"
# tokenizer.train(text, 256 + 1, verbose=True) # 256 are the byte tokens, then do 1 merges
print(tokenizer.encode(text))
# [258, 100, 258, 97, 99]
print(tokenizer.decode([258, 100, 258, 97, 99]))
# aaabdaaabac
tokenizer.save("toy")
# writes two files: toy.model (for loading) and toy.vocab (for viewing)

# %%
text = "hello123!!!? (안녕하세요!) 😉"

# tiktoken
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
print(enc.encode(text))
# [15339, 4513, 12340, 30, 320, 31495, 230, 75265, 243, 92245, 16715, 57037]

# ours
from minbpe import GPT4Tokenizer
tokenizer = GPT4Tokenizer()
print(tokenizer.encode(text))
# [15339, 4513, 12340, 30, 320, 31495, 230, 75265, 243, 92245, 16715, 57037]

# %%
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("_日本毛片免费视频观看"))
assert enc.decode(enc.encode("微信公众号天天中彩票 是什么意思")) == "微信公众号天天中彩票 是什么意思"
enc = tiktoken.encoding_for_model("gpt-4o")
assert enc.decode(enc.encode("hello world")) == "hello world"
