import json
from multiprocessing import Pool
import functools
from util import *


class InputExample(object):
    def __init__(self, text, en_pair_list, re_list, ex):
        self.text = text
        self.en_pair_list = en_pair_list
        self.re_list = re_list
        self.ex = ex


class InputFeatures(object):
    def __init__(self,
                 input_ids,
                 input_mask,
                 input_tokens=None,
                 en_table=None,
                 en_mask=None,
                 ht_table=None,
                 ht_mask=None,
                 ex=None,
                 ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_tokens = input_tokens
        self.en_table = en_table
        self.en_mask = en_mask
        self.ht_table = ht_table
        self.ht_mask = ht_mask
        self.ex = ex


def json2list(data_dir, data_sign):
    examples = []
    data_path = os.path.join(data_dir, data_sign + ".json")
    with open(data_path, "r", encoding='utf-8') as f:
        data = json.load(f)
        for sample in data:
            text = sample['text']
            en_pair_list = []
            re_list = []
            for triple in sample['triple_list']:
                en_pair_list.append([triple[0], triple[-1]])
                re_list.append(triple[1])
            example = InputExample(text=text, en_pair_list=en_pair_list, re_list=re_list, ex=sample)
            examples.append(example)
    return examples


def read_examples(data, args, tokenizer, data_sign, rel2idx):
    max_len = args.max_len
    input_tokens = tokenizer.tokenize(data.text)
    input_ids = tokenizer.tokens_to_ids(input_tokens)
    input_mask = [1] * len(input_ids)
    if len(input_tokens) > max_len:
        input_tokens = input_tokens[:max_len]
        input_ids = input_ids[:max_len]
        input_mask = input_mask[:max_len]
    n = len(input_ids)
    if type(input_ids) is np.ndarray:
        input_ids = input_ids.tolist()
    if data_sign == 'train':
        en_table = np.zeros([n, n])
        ht_table = np.zeros([n, n, len(rel2idx)])
        for en_pair, rel in zip(data.en_pair_list, data.re_list):
            s, o = en_pair[0], en_pair[-1]
            s = tokenizer.encode(s)[0][1:-1]
            p = rel2idx[rel]
            o = tokenizer.encode(o)[0][1:-1]
            s_idx = search(s, input_ids)
            o_idx = search(o, input_ids)
            if s_idx != -1 and o_idx != -1:
                s = (s_idx, s_idx + len(s) - 1)
                o = (o_idx, o_idx + len(o) - 1)
                s1, s2 = s
                o1, o2 = o
                en_table[s1, s2] = 1
                en_table[o1, o2] = 1
                if s1 == s2 and o1 == o2:
                    if ht_table[s1, o1, p] == 0:
                        ht_table[s1, o1, p] = 3
                    elif ht_table[s1, o1, p] == 1:
                        ht_table[s1, o1, p] = 4
                    elif ht_table[s1, o1, p] == 2:
                        ht_table[s1, o1, p] = 5
                else:
                    if ht_table[s1, o1, p] == 0:
                        ht_table[s1, o1, p] = 1
                    elif ht_table[s1, o1, p] == 3:
                        ht_table[s1, o1, p] = 4
                    if ht_table[s2, o2, p] == 0:
                        ht_table[s2, o2, p] = 2
                    elif ht_table[s2, o2, p] == 3:
                        ht_table[s2, o2, p] = 5
            en_mask = np.triu(np.ones((len(input_tokens), len(input_tokens))), k=0)
            ht_mask = np.ones([len(input_tokens), len(input_tokens), len(rel2idx)])
            en_mask[0, :] = 0
            en_mask[-1, :] = 0
            en_mask[:, 0] = 0
            en_mask[:, -1] = 0
            ht_mask[0, :, :] = 0
            ht_mask[-1, :, :] = 0
            ht_mask[:, 0, :] = 0
            ht_mask[:, -1, :] = 0
        return InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            en_table=en_table,
            en_mask=en_mask,
            ht_table=ht_table,
            ht_mask=ht_mask
        )
    else:
        return InputFeatures(
            input_tokens=input_tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            ex=data.ex
        )


def mul_process(args, tokenizer, data_dir, data_sign, rel2idx):
    data = json2list(data_dir, data_sign)
    with Pool(10) as p:
        convert_func = functools.partial(read_examples, args=args, tokenizer=tokenizer, data_sign=data_sign,
                                         rel2idx=rel2idx)
        features = p.map(func=convert_func, iterable=data)
    return features
