from transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup
from bert4keras.tokenizers import Tokenizer
from model import SOIRP
from util import *
from tqdm import tqdm
import os
import torch.nn as nn
import torch
from transformers.modeling_bert import BertConfig
import json
from transformers import BertTokenizer
from dataloader import CustomDataLoader


def train(args):
    set_seed()
    try:
        torch.cuda.set_device(int(args.cuda_id))
    except:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
    output_path = os.path.join(args.base_path, args.dataset, "output", args.file_id)
    rel2id_path = os.path.join(args.base_path, args.dataset, "rel2id.json")
    test_pred_path = os.path.join(output_path, "test_pred.json")
    dev_pred_path = os.path.join(output_path, "dev_pred.json")
    log_path = os.path.join(output_path, "log.txt")
    id2predicate, predicate2id = json.load(open(rel2id_path))
    bert4tokenizer = Tokenizer(args.bert_vocab_path)
    tokenizer = BertTokenizer(vocab_file=args.bert_vocab_path, do_lower_case=False)
    config = BertConfig.from_pretrained(args.bert_config_path)
    config.num_p = len(id2predicate)
    config.rounds = args.rounds
    config.fix_bert_embeddings = args.fix_bert_embeddings
    config.en_label = 2
    config.ht_label = 6
    loss_weight = args.loss_weight
    train_model = SOIRP.from_pretrained(pretrained_model_name_or_path=args.bert_model_path, config=config)
    train_model.to("cuda")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print_config(args)
    dataloader = CustomDataLoader(args)
    train_loader = dataloader.get_dataloader(data_sign='train')
    dev_loader = dataloader.get_dataloader(data_sign='dev')
    test_loader = dataloader.get_dataloader(data_sign='test')
    t_total = len(train_loader) * args.num_train_epochs
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in train_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in train_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.min_num)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup * t_total, num_training_steps=t_total
    )
    best_f1 = -1.0
    step = 0

    def crossloss(pre, target, mask=None):
        pre = pre.reshape([-1, pre.shape[-1]])
        target = target.reshape([-1])
        mask = mask.reshape([-1])
        crossentropy = nn.CrossEntropyLoss(reduction="none")
        loss = crossentropy(pre, target)
        loss = (loss * mask).sum()
        return loss

    for epoch in range(args.num_train_epochs):
        train_model.train()
        epoch_loss = 0
        with tqdm(total=train_loader.__len__(), desc="train") as t:
            for i, batch in enumerate(train_loader):
                batch = [torch.tensor(d).to("cuda") for d in batch]
                batch_input_ids, batch_input_mask, batch_en_table, batch_en_mask, batch_ht_table, batch_ht_mask = batch
                en_table, ht_table = train_model(batch_input_ids, batch_input_mask)
                en_table_loss = crossloss(en_table, batch_en_table.long(), batch_en_mask)
                ht_table_loss = crossloss(ht_table, batch_ht_table.long(), batch_ht_mask)
                loss = 2 * loss_weight * en_table_loss + (2 - loss_weight * 2) * ht_table_loss
                loss.backward()
                step += 1
                epoch_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                train_model.zero_grad()
                t.set_postfix(loss='{:.4f}'.format(loss.cpu().item()),
                              en_table_loss='{:.4f}'.format(en_table_loss.cpu().item()),
                              ht_table_loss='{:.4f}'.format(ht_table_loss.cpu().item()))
                t.update(1)
        f1, precision, recall = evaluate(args, tokenizer, train_model, dev_loader, dev_pred_path, id2predicate,
                                         bert4tokenizer)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(train_model.state_dict(), os.path.join(output_path, WEIGHTS_NAME))
        epoch_loss = epoch_loss / len(train_loader)
        with open(log_path, "a", encoding="utf-8") as f:
            print("epoch:%d\tloss:%f\tf1:%f\tprecision:%f\trecall:%f\tbest_f1:%f\t" % (
                int(epoch), epoch_loss, f1, precision, recall, best_f1), file=f)
    train_model.load_state_dict(torch.load(os.path.join(output_path, WEIGHTS_NAME), map_location="cuda"))
    f1, precision, recall = evaluate(args, tokenizer, train_model, test_loader, test_pred_path, id2predicate,
                                     bert4tokenizer)
    with open(log_path, "a", encoding="utf-8") as f:
        print("test： f1:%f\tprecision:%f\trecall:%f" % (f1, precision, recall), file=f)


def extract_spoes(batch, args, tokenizer, model, id2predicate, bert4tokenizer):
    def get_pred_spo(en_table, ht_table, input_ids, input_tokens):
        B, L, _, R, _ = ht_table.shape
        res = []
        for i in range(B):
            res.append([])
        en_table = en_table.argmax(axis=-1)
        ht_table = ht_table.argmax(axis=-1)
        all_loc = np.where(ht_table != 0)
        res_dict = []
        for i in range(B):
            res_dict.append([])
        for i in range(len(all_loc[0])):
            token_n = len(input_tokens[all_loc[0][i]])
            if token_n - 2 <= all_loc[1][i] \
                    or token_n - 2 <= all_loc[2][i] \
                    or 0 in [all_loc[1][i], all_loc[2][i]]:
                continue
            res_dict[all_loc[0][i]].append([all_loc[1][i], all_loc[2][i], all_loc[3][i]])
        for i in range(B):
            for l1, l2, r in res_dict[i]:
                if ht_table[i, l1, l2, r] == 3:
                    if en_table[i, l1, l1] == 1 and en_table[i, l2, l2] == 1:
                        res[i].append([l1, l1, r, l2, l2])
                        continue
                elif ht_table[i, l1, l2, r] == 4 or ht_table[i, l1, l2, r] == 5:
                    if en_table[i, l1, l1] == 1 and en_table[i, l2, l2] == 1:
                        res[i].append([l1, l1, r, l2, l2])
                if ht_table[i, l1, l2, r] == 1 or ht_table[i, l1, l2, r] == 4:
                    for l1_, l2_, r_ in res_dict[i]:
                        if r == r_ and (ht_table[i, l1_, l2_, r_] == 2 or ht_table[i, l1_, l2_, r_] == 5) and en_table[
                            i, l1, l1_] == 1 and en_table[i, l2, l2_] == 1 \
                                and l1_ >= l1 and l2_ >= l2:
                            res[i].append([l1, l1_, r, l2, l2_])
                            break
        return res

    _, _, batch_input_tokens, batch_ex = batch
    batch_input_ids, batch_input_mask = [torch.tensor(d).to("cuda") for d in batch[:-2]]
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.to("cuda")
    model.eval()
    with torch.no_grad():
        en_table, ht_table = model(batch_input_ids, batch_input_mask)
        en_table = en_table.cpu().detach().numpy()
        ht_table = ht_table.cpu().detach().numpy()
    res_id = get_pred_spo(en_table, ht_table, batch_input_ids, batch_input_tokens)
    batch_spo = [[] for _ in range(len(batch_ex))]
    for b, ex in enumerate(batch_ex):
        text = ex["text"]
        tokens = batch_input_tokens[b]
        mapping = bert4tokenizer.rematch(text, tokens)
        for sh, st, r, oh, ot in res_id[b]:
            s = (mapping[sh][0], mapping[st][-1])
            o = (mapping[oh][0], mapping[ot][-1])
            batch_spo[b].append(
                (text[s[0]:s[1] + 1], id2predicate[str(r)], text[o[0]:o[1] + 1])
            )
    return batch_spo


def evaluate(args, tokenizer, model, dataloader, evl_path, id2predicate, bert4tokenizer):
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open(evl_path, 'w', encoding='utf-8')
    pbar = tqdm()
    for batch in dataloader:
        batch_ex = batch[-1]
        batch_spo = extract_spoes(batch, args, tokenizer, model, id2predicate, bert4tokenizer)
        for i, ex in enumerate(batch_ex):
            R = set(batch_spo[i])
            T = set([(item[0], item[1], item[2]) for item in ex['triple_list']])
            X += len(R & T)
            Y += len(R)
            Z += len(T)
            f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
            pbar.update()
            pbar.set_description(
                'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
            )
            s = json.dumps({
                'text': ex['text'],
                'triple_list': list(T),
                'triple_list_pred': list(R),
                'new': list(R - T),
                'lack': list(T - R),
            }, ensure_ascii=False, indent=4)
            f.write(s + '\n')
    pbar.close()
    f.close()
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


def test(args):
    try:
        torch.cuda.set_device(int(args.cuda_id))
    except:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
    output_path = os.path.join(args.base_path, args.dataset, "output", args.file_id)
    rel2id_path = os.path.join(args.base_path, args.dataset, "rel2id.json")
    id2predicate, predicate2id = json.load(open(rel2id_path))
    bert4tokenizer = Tokenizer(args.bert_vocab_path)
    tokenizer = BertTokenizer(vocab_file=args.bert_vocab_path, do_lower_case=False)
    config = BertConfig.from_pretrained(args.bert_config_path)
    config.num_p = len(id2predicate)
    config.rounds = args.rounds
    config.fix_bert_embeddings = args.fix_bert_embeddings
    config.en_label = 2
    config.ht_label = 6
    train_model = SOIRP.from_pretrained(pretrained_model_name_or_path=args.bert_model_path, config=config)
    train_model.to("cuda")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print_config(args)
    dataloader = CustomDataLoader(args)
    sign = "test"
    test_pred_path = os.path.join(output_path, sign + ".json")
    test_loader = dataloader.get_dataloader(data_sign=sign)
    train_model.load_state_dict(torch.load(os.path.join(output_path, WEIGHTS_NAME), map_location="cuda"))
    f1, precision, recall = evaluate(args, tokenizer, train_model, test_loader, test_pred_path, id2predicate,
                                     bert4tokenizer)
    print("f1:%f,precision:%f, recall:%f" % (f1, precision, recall))
