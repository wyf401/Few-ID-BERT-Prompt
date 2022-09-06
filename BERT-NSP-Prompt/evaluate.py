import argparse
import os
from collections import Counter
import numpy as np
import torch
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertConfig

from model import Model
from utils.data_utils import NluDataset, glue_processor, prepare_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(model, data_raw, id_to_label, tokenizer, mode='dev'):
    slot_label_list = id_to_label['slot_labels']
    intent_label_list = id_to_label['intent_labels']

    model.eval()
    test_data = NluDataset(data_raw)
    test_dataloader = DataLoader(test_data, batch_size=len(intent_label_list), collate_fn=test_data.collate_fn)

    joint_all = 0
    joint_correct = 0
    s_preds = []
    s_labels = []
    i_preds = []
    i_labels = []

    predicted_masked_tokens = []
    epoch_pbar = tqdm(test_dataloader, desc="Evaluation", disable=False)
    for step, batch in enumerate(test_dataloader):
        batch = [b.to(device) if not isinstance(b, int) else b for b in batch]
        input_ids, segment_ids, input_mask, intent_id = batch
        with torch.no_grad():
            # intent_output, slot_output = model(input_ids, segment_ids, input_mask, prompt_idx)
            seq_relationship_score = model(input_ids, segment_ids, input_mask)
        # intent_evaluate
        intent_output = seq_relationship_score.argmax()
        intent_output = intent_output.tolist()
        # print(intent_output)
        # print(data_raw[step * len(intent_label_list)].words)

        intent_label = intent_id.tolist().index(1)

        i_preds.append(intent_output)
        i_labels.append(intent_label)

        epoch_pbar.update(1)
    epoch_pbar.close()

    class_report_str = classification_report(i_labels, i_preds, target_names=intent_label_list, labels=[i for i in range(len(intent_label_list))])
    class_report = reconstruct_class_report(class_report_str)
    print(class_report)
    # return res
    i_preds_raw = [intent_label_list[x] for x in i_preds]
    i_labels_raw = [intent_label_list[x] for x in i_labels]
    sentences = [x.words for idx,x in enumerate(data_raw) if not idx % len(intent_label_list)]
    # write_prediction_to_file(sentences, i_preds_raw, i_labels_raw)

    eval_res = {"intent_acc": cal_acc(i_preds_raw, i_labels_raw)}
    print("%s dataset evaluate results: %s" %(mode, eval_res))
    return eval_res, class_report

def reconstruct_class_report(class_report_str):
    class_report = {}
    all_lines = class_report_str.split('\n')
    for line in all_lines[2:-5]:
        line = line.strip().split()
        class_report[line[0]] = [float(line[-2]), int(line[-1])]
    return class_report

def write_prediction_to_file(sentences, predicted_masked_tokens, i_labels_raw, filename='preds'):
    res = []
    bad = []
    a_to_b_bad_case = []
    for s,t,l in zip(sentences, predicted_masked_tokens, i_labels_raw):
        res.append("input sentence: %s \n" % s)
        res.append("predicted intent: %s \n" %t)
        res.append("intent label: %s \n\n" %l)
        if t != l:
            bad.append("input sentence: %s \n" % s)
            bad.append("predicted intent: %s \n" % t)
            bad.append("intent label: %s \n\n" % l)
            a_to_b_bad_case.append(l+'_to_'+t)
    # print(Counter(a_to_b_bad_case))
    with open(filename,'w',encoding='utf-8') as f:
        f.writelines(res)

    with open('bad_case','w',encoding='utf-8') as f:
        f.writelines(bad)

def cal_acc(preds, labels):
    acc = sum([1 if p == l else 0 for p, l in zip(preds, labels)]) / len(labels)
    return acc


def align_predictions(preds, slot_ids, id_to_label):
    aligned_labels = []
    aligned_preds = []
    for p, l in zip(preds, slot_ids):
        if l != -100:
            aligned_preds.append(id_to_label[p])
            aligned_labels.append(id_to_label[l])
    return aligned_preds, aligned_labels


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    # Init
    set_seed(args.seed)
    processor = glue_processor[args.task_name.lower()]
    tokenizer = BertTokenizer(args.vocab_path, do_lower_case=True)

    # Data
    dev_examples = processor.get_dev_examples(args.data_dir)
    test_examples = processor.get_test_examples(args.data_dir)
    labels = processor.get_labels(args.data_dir)
    prompt_sent_list = processor.get_prompt_sent(args.data_dir)
    # dev_data_raw = prepare_data(dev_examples, args.max_seq_len, tokenizer, labels, prompt_sent_list)
    test_data_raw = prepare_data(test_examples, args.max_seq_len, tokenizer, labels, prompt_sent_list)

    # Model
    model_config = BertConfig.from_json_file(args.bert_config_path)
    model_config.use_crf = args.use_crf
    model_config.dropout = args.dropout
    model_config.num_intent = len(labels['intent_labels'])
    model_config.num_slot = len(labels['slot_labels'])
    model = Model.from_pretrained(config=model_config, pretrained_model_name_or_path=args.model_ckpt_path)
    # ckpt = torch.load(args.model_ckpt_path, map_location='cpu')
    # model.load_state_dict(ckpt, strict=False)
    model.to(device)
    # evaluate(model, dev_data_raw, labels,tokenizer, 'dev')
    evaluate(model, test_data_raw, labels, tokenizer,'test')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--task_name", default='nlu', type=str)
    parser.add_argument("--data_dir", default='data/snips/', type=str)
    parser.add_argument("--model_path", default='assets/', type=str)

    parser.add_argument("--model_ckpt_path", default='assets/pytorch_model.bin', type=str)
    parser.add_argument("--use_crf", default=False, type=bool)
    parser.add_argument("--max_seq_len", default=80, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    args = parser.parse_args()
    args.vocab_path = os.path.join(args.model_path, 'vocab.txt')
    args.bert_config_path = os.path.join(args.model_path, 'config.json')
    print(args)
    main(args)
