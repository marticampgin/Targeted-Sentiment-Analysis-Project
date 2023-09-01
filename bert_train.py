import os
import numpy as np
import torch
import transformers

from argparse import ArgumentParser
from read_dataset import load_prepare_data, BERTDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from modeling_norbert import NorbertForTokenClassification
from evaluate_tsa import evaluate_model
from practical_functions import combined_train_dataset


class CollateFunctor:
    """
    Simple collate functor to pad sequences,
    labels and attention mask
    """

    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, samples):
        seq_vectors = []
        seq_labels = []
        att_masks = []
        orig_inputs = []

        for sample in sorted(samples, key=lambda x: len(x["sent"]), reverse=True):
            seq_vectors.append(sample["sent"])
            seq_labels.append(sample["labels"])
            att_masks.append(sample["att_mask"])
            orig_inputs.append((sample["orig_inputs"]))

        padded_sequences_vectors = pad_sequence(seq_vectors,
                                                batch_first=True,
                                                padding_value=self.pad_idx["seq"])

        padded_sequences_labels = pad_sequence(seq_labels,
                                               batch_first=True,
                                               padding_value=self.pad_idx["labels"])

        padded_att_masks = pad_sequence(att_masks,
                                        batch_first=True,
                                        padding_value=self.pad_idx["att_mask"])

        return {"sent": padded_sequences_vectors,
                "labels": padded_sequences_labels,
                "att_mask": padded_att_masks,
                "orig_inputs": orig_inputs}


def seed_everything(args):
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def train(model, train_iter, optimizer, device):
    model.train()
    for batch in train_iter:
        optimizer.zero_grad()
        sent = batch["sent"].to(device)
        labels = batch["labels"].to(device)
        att_mask = batch["att_mask"].to(device)
        out = model(sent, attention_mask=att_mask, labels=labels)
        loss = out.loss
        loss.backward()
        optimizer.step()


def main():
    parser = ArgumentParser()
    parser.add_argument("--tsa_folder", default=r"data/tsa_conll")
    parser.add_argument("--transformer_path", default=r"model/norbert3_base")
    parser.add_argument("--metadata", default=r"metadata.json")
    parser.add_argument("--train_dom", action="store", type=str, required=True)
    parser.add_argument("--test_dom", action="store", type=str, required=True)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", action="store", type=int, default=777)
    parser.add_argument("--batch_size", action="store", type=int, default=16)
    parser.add_argument("--epochs", action="store", type=int, default=8)
    parser.add_argument("--classes", action="store", type=int, default=5)
    parser.add_argument("--lr", action="store", type=float, default=3e-5)
    parser.add_argument("--freeze",
                        action="store",
                        help="1 to freeze the model, 0 to let it fine-tune",
                        type=int,
                        default=0)

    args = parser.parse_args()

    seed_everything(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid warnings

    # Label and id mappings
    label2id = {'O': 0, 'B-targ-Positive': 1,
                'I-targ-Positive': 2,
                'B-targ-Negative': 3,
                'I-targ-Negative': 4, }

    id2label = {val: key for key, val in label2id.items()}

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.transformer_path)

    def tokenize_and_align_labels(examples):
        """
        Function that tokenizes input and aligns labels.
        :param examples: huggingface dataset
        :return:
        """
        tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word
            previous_word_index = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_index:  # Only label the first token of a given word
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_index = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # Using batched = True to speed up the process
    train_domains = [domain for domain in args.train_dom.split("_")]
    test_domains = [args.test_dom]

    dataset = load_prepare_data(args, label2id)
    train_ds, test_ds = combined_train_dataset(dataset, train_domains, test_domains,  args.seed, ratio=0.30)

    train_tokenized_ds = train_ds.map(tokenize_and_align_labels, batched=True)
    test_tokenized_ds = test_ds.map(tokenize_and_align_labels, batched=True)

    # Loading the model
    model = NorbertForTokenClassification.from_pretrained(args.transformer_path,
                                                          num_labels=args.classes,
                                                          id2label=id2label,
                                                          label2id=label2id).to(device)

    # Last two tensor are classifier parameters
    if args.freeze == 1:
        for param in list(model.parameters())[:-2]:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)

    # Extracting input ids, labels and attention masks
    train_data, train_labels = train_tokenized_ds["input_ids"], train_tokenized_ds["labels"]
    train_attention_mask = train_tokenized_ds["attention_mask"]

    test_data, test_labels = test_tokenized_ds["input_ids"], test_tokenized_ds["labels"]
    test_attention_mask, orig_inputs = test_tokenized_ds["attention_mask"], test_tokenized_ds["tokens"]

    train_dataset = BERTDataset(train_data, train_labels, train_attention_mask)
    test_dataset = BERTDataset(test_data, test_labels, test_attention_mask, orig_inputs)

    # Padding idx. for input ids, labels and att. masks
    pad_idx = {"seq": 3, "labels": -100, "att_mask": 0}

    train_iter = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            num_workers=2,
                            collate_fn=CollateFunctor(pad_idx))

    test_iter = DataLoader(test_dataset,
                           batch_size=args.batch_size,
                           num_workers=2,
                           collate_fn=CollateFunctor(pad_idx))

    for epoch in range(args.epochs):
        train(model, train_iter, optimizer, device)
        tr_prec, tr_rec, tr_f1 = evaluate_model(model, train_iter, device, id2label)
        test_prec, test_rec, test_f1 = evaluate_model(model, test_iter, device, id2label, tokenizer, epoch, test_domains[-1], save_sents=True)

        print(f"Epoch: {epoch}")
        print(f"Train scores:  recall: {tr_rec}\tprecision: {tr_prec}\t f1: {tr_f1}")
        print(f"Valid. scores: recall: {test_rec}\tprecision: {test_prec}\t f1: {test_f1}", end="\n\n")

    # path_to_save = "saved_model/"
    # model.save_pretrained(path_to_save, from_pt=True)


if __name__ == "__main__":
    main()
