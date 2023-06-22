import json
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from bert4keras.tokenizers import Tokenizer
from dataloader_utils import mul_process
from util import *


class FeatureDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


class CustomDataLoader(object):
    def __init__(self, args):
        self.args = args
        self.train_batch_size = args.train_batch_size
        self.val_batch_size = args.val_batch_size
        self.test_batch_size = args.test_batch_size
        self.data_cache = args.data_cache
        self.data_dir = os.path.join(args.base_path, args.dataset)
        self.max_seq_length = args.max_len
        self.tokenizer = Tokenizer(args.bert_vocab_path)

    @staticmethod
    def collate_fn_train(features):
        input_ids = [f.input_ids for f in features]
        input_mask = [f.input_mask for f in features]
        en_table = [f.en_table for f in features]
        en_mask = [f.en_mask for f in features]
        ht_table = [f.ht_table for f in features]
        ht_mask = [f.ht_mask for f in features]
        input_ids = sequence_padding(input_ids)
        input_mask = sequence_padding(input_mask)
        en_table = mat_padding(en_table)
        en_mask = mat_padding(en_mask)
        ht_table = mat_padding(ht_table)
        ht_mask = mat_padding(ht_mask)
        tensors = [input_ids, input_mask, en_table, en_mask, ht_table, ht_mask]
        return tensors

    @staticmethod
    def collate_fn_test(features):
        input_ids = [f.input_ids for f in features]
        input_mask = [f.input_mask for f in features]
        input_tokens = [f.input_tokens for f in features]
        ex = [f.ex for f in features]
        input_ids = sequence_padding(input_ids)
        input_mask = sequence_padding(input_mask)
        input_tokens = [np.array(i) for i in input_tokens]
        tensors = [input_ids, input_mask, input_tokens, ex]
        return tensors

    def get_features(self, data_sign):
        print("=*=" * 10)
        print("Loading {} data...".format(data_sign))
        rel2id_path = os.path.join(self.data_dir, "rel2id.json")
        cache_path = os.path.join(self.data_dir, "{}.cache.{}".format(data_sign, str(self.max_seq_length)))
        if os.path.exists(cache_path) and self.data_cache:
            features = torch.load(cache_path)
        else:
            with open(rel2id_path, 'r', encoding='utf-8') as f_re:
                rel2idx = json.load(f_re)[-1]
            if data_sign in ("train", "dev", "test", 'EPO', 'SEO', 'Normal', '1', '2', '3', '4', '5'):
                features = mul_process(self.args, self.tokenizer, self.data_dir, data_sign, rel2idx)
            else:
                raise ValueError("please notice that the data can only be train/dev/test!!")
            if self.data_cache:
                torch.save(features, cache_path)
        return features

    def get_dataloader(self, data_sign):
        features = self.get_features(data_sign)
        dataset = FeatureDataset(features)
        print(f"{len(features)} {data_sign} data loaded!")
        print("=*=" * 10)
        if data_sign == "train":
            datasampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.train_batch_size,
                                    collate_fn=self.collate_fn_train, num_workers=12)
        elif data_sign == "dev":
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.val_batch_size,
                                    collate_fn=self.collate_fn_test, num_workers=6)
        elif data_sign in ("test", "pseudo", 'EPO', 'SEO', 'SOO', 'Normal', '1', '2', '3', '4', '5'):
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.test_batch_size,
                                    collate_fn=self.collate_fn_test, num_workers=6)
        else:
            raise ValueError("please notice that the data can only be train/dev/test !!")
        return dataloader
