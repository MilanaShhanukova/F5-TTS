from __future__ import annotations

import os
import random
from collections import defaultdict
from importlib.resources import files

import torch
from torch.nn.utils.rnn import pad_sequence

import jieba
from pypinyin import lazy_pinyin, Style


# seed everything


def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# helpers


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


# tensor helpers


def lens_to_mask(t: int["b"], length: int | None = None) -> bool["b n"]:  # noqa: F722 F821
    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]


def mask_from_start_end_indices(seq_len: int["b"], start: int["b"], end: int["b"]):  # noqa: F722 F821
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask


def mask_from_frac_lengths(seq_len: int["b"], frac_lengths: float["b"]):  # noqa: F722 F821
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)


def maybe_masked_mean(t: float["b n d"], mask: bool["b n"] = None) -> float["b d"]:  # noqa: F722
    if not exists(mask):
        return t.mean(dim=1)

    t = torch.where(mask[:, :, None], t, torch.tensor(0.0, device=t.device))
    num = t.sum(dim=1)
    den = mask.float().sum(dim=1)

    return num / den.clamp(min=1.0)


# simple utf-8 tokenizer, since paper went character based
def list_str_to_tensor(text: list[str], padding_value=-1) -> int["b nt"]:  # noqa: F722
    list_tensors = [torch.tensor([*bytes(t, "UTF-8")]) for t in text]  # ByT5 style
    text = pad_sequence(list_tensors, padding_value=padding_value, batch_first=True)
    return text

def char_to_num(text: str, vocab_char_map):
    tokens = [vocab_char_map.get(c, 0) for c in text]
    tokens = torch.tensor(tokens)
    return tokens


def transform_text_to_ipa(text,
                        vocab_char_map):
    # tokenize including the multi-char ipa tokens
    tokens = []
    i = 0
    if isinstance(text, list):
        text = "".join(text)

    while i < len(text):
        # Check for multi-character tokens (longest match first)
        match_found = False
        for symbol in sorted(vocab_char_map.keys(), key=len, reverse=True):
            if text[i:i+len(symbol)] == symbol:
                tokens.append(vocab_char_map[symbol])
                i += len(symbol)
                match_found = True
                break
        if not match_found:
            tokens.append(0)
            i += 1
    tokens = torch.tensor(tokens)
    return tokens


def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    tokenizer_type: str,
    padding_value = -1
) -> int['b nt']:
    if tokenizer_type == "char":
        list_idx_tensors = [char_to_num(t, vocab_char_map) for t in text]  # pinyin or char style
    elif tokenizer_type == "ipa":
        list_idx_tensors = [transform_text_to_ipa(t, vocab_char_map) for t in text]

    text = pad_sequence(list_idx_tensors, padding_value = padding_value, batch_first = True)
    return text


def split_text_to_ipa(text,
                    vocab_char_map):
    # tokenize including the multi-char ipa tokens
    tokens = []
    i = 0
    while i < len(text):
        # Check for multi-character tokens (longest match first)
        match_found = False
        for symbol in sorted(vocab_char_map.keys(), key=len, reverse=True):
            if text[i:i+len(symbol)] == symbol:
                tokens.append(symbol)
                i += len(symbol)
                match_found = True
                break
        if not match_found:
            # tokens.append(0)
            i += 1
    # tokens = torch.tensor(tokens)
    return tokens


# Get tokenizer

def load_tokenizer(vocab_file_path, golden_vocab={}):
    with open(vocab_file_path, "r") as f:
        all_symbols = f.read().splitlines()

        for i, char in enumerate(all_symbols):
            if char not in golden_vocab and char not in golden_vocab:
                golden_vocab[char] = len(golden_vocab)

    import json
    with open('vocab_mapping.json', 'w', encoding='utf-8') as json_file:
        json.dump(golden_vocab, json_file, ensure_ascii=False, indent=4)

    return golden_vocab, len(golden_vocab)


def get_tokenizer(dataset_name, tokenizer: str = "pinyin"):
    """
    tokenizer   - "pinyin" do g2p for only chinese characters, need .txt vocab_file
                - "char" for char-wise tokenizer, need .txt vocab_file
                - "byte" for utf-8 tokenizer
                - "custom" if you're directly passing in a path to the vocab.txt you want to use
    vocab_size  - if use "pinyin", all available pinyin types, common alphabets (also those with accent) and symbols
                - if use "char", derived from unfiltered character & symbol counts of custom dataset
                - if use "byte", set to 256 (unicode byte range)
    """
    if os.path.exists(dataset_name):
        vocab_char_map, vocab_size = load_tokenizer(dataset_name)
        return vocab_char_map, vocab_size

    if tokenizer in ["pinyin", "char", "ipa"]:
        tokenizer_path = os.path.join(files("f5_tts").joinpath("../../data"), f"{dataset_name}_{tokenizer}/vocab.txt")
        dataset_path = os.path.abspath(tokenizer_path)
        vocab_char_map, vocab_size = load_tokenizer(dataset_path)
        assert vocab_char_map[" "] == 0, "make sure space is of idx 0 in vocab.txt, cuz 0 is used for unknown char"

    elif tokenizer == "byte":
        vocab_char_map = None
        vocab_size = 256

    elif tokenizer == "custom":
        vocab_char_map, vocab_size = load_tokenizer(dataset_path)

    return vocab_char_map, vocab_size


# convert char to pinyin

def convert_char_to_ipa(text_list, vocab_char_map):
    texts = [split_text_to_ipa(text, vocab_char_map) for text in text_list]
    return texts


def convert_char_to_pinyin(text_list, polyphone=True):
    final_text_list = []
    god_knows_why_en_testset_contains_zh_quote = str.maketrans(
        {"“": '"', "”": '"', "‘": "'", "’": "'"}
    )  # in case librispeech (orig no-pc) test-clean
    custom_trans = str.maketrans({";": ","})  # add custom trans here, to address oov
    for text in text_list:
        char_list = []
        text = text.translate(god_knows_why_en_testset_contains_zh_quote)
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # if pure chinese characters
                seg = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for c in seg:
                    if c not in "。，、；：？！《》【】—…":
                        char_list.append(" ")
                    char_list.append(c)
            else:  # if mixed chinese characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    else:
                        if c not in "。，、；：？！《》【】—…":
                            char_list.append(" ")
                            char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                        else:  # if is zh punc
                            char_list.append(c)
        final_text_list.append(char_list)

    return final_text_list


# filter func for dirty data with many repetitions


def repetition_found(text, length=2, tolerance=10):
    pattern_count = defaultdict(int)
    for i in range(len(text) - length + 1):
        pattern = text[i : i + length]
        pattern_count[pattern] += 1
    for pattern, count in pattern_count.items():
        if count > tolerance:
            return True
    return False


def expand_model_embeddings(ckpt_path: str, new_ckpt_path: str, vocab_size: int):
    seed = 666
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ema_sd = ckpt.get("ema_model_state_dict", {})
    embed_key_ema = "ema_model.transformer.text_embed.text_embed.weight"
    old_embed_ema = ema_sd[embed_key_ema]
    vocab_old = old_embed_ema.size(0)
    embed_dim = old_embed_ema.size(1)

    vocab_size += 1 # include the filler token

    def expand_embeddings(old_embeddings):
        new_embeddings = torch.zeros((vocab_size, embed_dim))
        new_embeddings[:vocab_old] = old_embeddings
        
        add_tokens = vocab_size - vocab_old

        new_embeddings[vocab_old:] = torch.randn((add_tokens, embed_dim))
        return new_embeddings
    
    if vocab_size > vocab_old:
        print(f"Expanding the tokens {new_ckpt_path}, from {vocab_old} to {vocab_size}")
        ema_sd[embed_key_ema] = expand_embeddings(ema_sd[embed_key_ema])
        torch.save(ckpt, new_ckpt_path)
        return new_ckpt_path
    return ckpt_path