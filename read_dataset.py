import torch
import os
import json
import random

from torch.utils.data import Dataset


def parse_conll(args, raw: str, sep="\t"):
    """Parses the norec-fine conll files with tab separator and sentence id"""
    metadata_file = open(args.metadata, encoding="utf8")
    metadata = json.load(metadata_file)

    doc_parsed = []  # One dict per sentence. meta, tokens and tags
    for sent in raw.strip().split("\n\n"):
        meta = ""
        doc_id = ""
        domain = ""
        tokens, tags = [], []
        for line in sent.split("\n"):
            if line.startswith("#") and "=" in line:
                meta = line.split("=")[-1]
                doc_id = meta[:6]  # first six digits
                domain = metadata[doc_id]["category"]
            else:
                elems = line.strip().split(sep)
                assert len(elems) == 2
                tokens.append(elems[0])
                tags.append(elems[1])
        assert len(meta) > 0
        doc_parsed.append({"idx": meta, "tokens": tokens, "tsa_tags": tags, "domain": domain})

    return doc_parsed


def load_prepare_data(args, mapping):
    """Loads, parses .conll file and returns a huggingface dataset"""
    splits = ["train", "dev", "test"]
    splits_combined = []  # for merging train, dev, and test
    for split in splits:
        path = os.path.join(args.tsa_folder, split + ".conll")
        with open(path, encoding="utf8") as rf:
            conll_txt = rf.read()
        sents = parse_conll(args, conll_txt)
        for sent in sents:
            sent["labels"] = [mapping[tag] for tag in sent["tsa_tags"]]
        splits_combined += sents

    random.seed(args.seed)
    random.shuffle(splits_combined)
    return splits_combined


class BERTDataset(Dataset):
    """
    Torch dataset for BERT-like models
    """
    def __init__(self, input_ids, labels, attention_mask, orig_inputs=None):
        self.input_ids = input_ids
        self.labels = labels
        self.attention_mask = attention_mask
        self.orig_inputs = orig_inputs

    def __getitem__(self, idx):
        sentence = self.input_ids[idx]
        labels = self.labels[idx]
        att_mask = self.attention_mask[idx]

        return {"sent": torch.LongTensor(sentence),
                "labels": torch.LongTensor(labels),
                "att_mask": torch.LongTensor(att_mask),
                "orig_inputs": self.orig_inputs}

    def __len__(self):
        return len(self.input_ids)



