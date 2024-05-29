"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.
"""

import regex as re
from .base import Tokenizer, get_stats, merge


# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
"""
   |符号表示“或”操作，若满足了多个匹配规则，则按照从左到右的顺序进行匹配
   - `'(?:[sdmt]|ll|ve|re)`：匹配以单引号开始，后面跟着s, d, m, t, ll, ve, re的字符串。这主要用于匹配缩写，如I'm, I've等。
   - ` ?\p{L}+`：匹配一个或多个任何种类的字母，前面可能有一个空格。
   - ` ?\p{N}+`：匹配一个或多个任何种类的数字，前面可能有一个空格。
   - ` ?[^\s\p{L}\p{N}]+`：匹配一个或多个非空格、非字母、非数字的字符，前面可能有一个空格。
   - `\s+(?!\S)`：匹配一个或多个空格，后面不是非空字符。
   - `\s+`：匹配一个或多个空格。

    在这里，“任何种类”是指Unicode字符集中的所有种类。Unicode是一种包含世界上大多数字符系统的编码标准。
    在正则表达式中，`\p{L}`和`\p{N}`是Unicode属性转义序列，它们分别匹配任何种类的字母和数字。
    - `\p{L}`：匹配任何种类的字母，包括拉丁字母、希腊字母、俄罗斯字母、阿拉伯字母、中文字符等。
    - `\p{N}`：匹配任何种类的数字，包括阿拉伯数字、罗马数字等。
    所以，当我们说“任何种类的字母或数字”，我们是指所有Unicode字符集中的字母或数字。
"""
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
"""
   |符号表示“或”操作，若满足了多个匹配规则，则按照从左到右的顺序进行匹配
   - `'(?i:[sdmt]|ll|ve|re)`：匹配以单引号开始，后面跟着s, d, m, t, ll, ve, re的字符串，不区分大小写。这主要用于匹配缩写，如I'm, I've等。
   - `[^\r\n\p{L}\p{N}]?+\p{L}+`：匹配一个或多个字母，前面可能有一个非换行、非字母、非数字的字符。
   - `\p{N}{1,3}`：匹配1到3个数字。
   - ` ?[^\s\p{L}\p{N}]++[\r\n]*`：匹配一个或多个非空格、非字母、非数字的字符，后面可能有一个或多个换行，前面可能有一个空格。
   - `\s*[\r\n]`：匹配一个换行，前面可能有多个空格。
   - `\s+(?!\S)`：匹配一个或多个空格，后面不是非空字符。
   - `\s+`：匹配一个或多个空格。
"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        # 编译正则表达式可以提高代码的效率，特别是当你需要多次使用同一个正则表达式时。
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # split the text up into text chunks
        # 查找text中所有匹配的子串。它返回一个包含所有匹配的子串的列表。
        # 如果在text中没有找到与GPT4_SPLIT_PATTERN匹配的子串，re.findall()函数将返回一个空列表
        # re.findall()函数会按照从左到右的顺序在字符串中查找匹配的子串。一旦找到一个匹配，它就会继续在剩余的字符串中查找下一个匹配。
        text_chunks = re.findall(self.compiled_pattern, text)

        # input text preprocessing
        # eg. list('cat'.encode("utf-8")) 值为 [99, 97, 116]
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
        for i in range(num_merges):
            # count the number of times every consecutive pair appears
            stats = {}
            for chunk_ids in ids:
                # passing in stats will update it in place, adding up counts
                get_stats(chunk_ids, stats)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        # given ids (list of integers), return Python string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        # 将一个字节序列列表连接成一个单一的字节序列
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            # ordinary：普通的
            return self.encode_ordinary(text)
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        # 在正则表达式中，括号()被用来创建捕获组，这样在使用re.split函数分割文本时，特殊标记也会被包含在分割后的结果中。
        # eg: 如果特殊标记列表是['$', '%']，并且文本是'I have $100 and 100%'，那么special_chunks将会是['I have ', '$', '100 and 100', '%', '']。
        # 若没有()，则special_chunks将会是['I have ', '100 and 100', '']。
        # re.escape用来确保特殊标记中的所有字符都被正确地视为字面字符，而不是正则表达式的运算符，如 * + 等
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids
