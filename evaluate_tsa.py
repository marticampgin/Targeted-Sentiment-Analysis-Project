from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from ner_eval import Evaluator
from collections import Counter
import torch
import matplotlib.pyplot as plt


def f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    score = 2 * (precision * recall) / (precision + recall)
    return score


def evaluateur(gold, predictions):
    """Predictions and gold are lists of lists with token labels as text"""

    labels = set([l for s in gold for l in s])
    labels.remove('O')  # remove 'O' label from evaluation
    labels = list(set([l[2:] for l in labels]))
    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))

    evaluator = Evaluator(gold, predictions, sorted_labels)
    results, results_agg = evaluator.evaluate()

    # print("F1 scores:")
    for entity in results_agg:
        prec = results_agg[entity]["strict"]["precision"]
        rec = results_agg[entity]["strict"]["recall"]
        # print(f"{entity}:\t{f1(prec, rec):.4f}")
    prec = results["strict"]["precision"]
    rec = results["strict"]["recall"]

    return prec, rec, f1(prec, rec)


def convert_prediction(predictions, labels, mapping):
    true_predictions = [
        [mapping[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [mapping[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return true_labels, true_predictions


def count_errors(labels, predictions, sents, epoch, domain):
    wrongly_classified_words = []
    wrongly_classified_tags = {}
    for seq_id in range(len(labels)):  # sequence
        sent = sents[seq_id].split()
        label_seq = labels[seq_id]
        pred_seq = predictions[seq_id]
        for word_id in range(len(sent)):
            prediction, gold, word = pred_seq[word_id], label_seq[word_id], sent[word_id]
            if prediction != gold:
                wrongly_classified_words.append(word)
                wrongly_classified_tags[word] = f"gold:{gold}_pred:{prediction}"

    most_common = Counter(wrongly_classified_words).most_common(10)
    with open(f"{epoch}_{domain}_mc_errs.txt", "w", encoding="utf8") as f:
        for word, count in most_common:
            f.write(word + ": " + str(count) + "\t" + wrongly_classified_tags[word] + "\n")


@torch.no_grad()
def evaluate_model(model, data_iter, device, mapping, tokenizer=None, epoch=None, domain=None, save_sents=False):
    """
    Function to evaluate the performance of
    contextualized word-embedding models
    """
    model.eval()
    overall_prec = 0
    overall_rec = 0
    overall_f1 = 0
    batches = len(data_iter)

    sents = []
    all_labels = []
    all_predictions = []

    for i, batch in enumerate(data_iter):
        sent = batch["sent"].to(device)
        labels = batch["labels"].to(device)
        att_mask = batch["att_mask"].to(device)

        if save_sents:
            for sample in sent:
                sample = tokenizer.decode(sample, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                sents.append(sample)

        out = model(sent, attention_mask=att_mask, labels=labels)

        predictions = out.logits
        predictions = predictions.argmax(dim=2).tolist()
        labels = labels.tolist()

        labels, predictions = convert_prediction(predictions, labels, mapping)
        all_labels.append(labels)
        all_predictions.append(predictions)

        prec, rec, f1_score = evaluateur(labels, predictions)
        overall_prec += prec
        overall_rec += rec
        overall_f1 += f1_score

    if save_sents:
        labels, predictions = [], []
        conf_mat_labels, conf_mat_preds = [], []

        for i in range(len(all_labels)):
            labels_batch = all_labels[i]
            preds_batch = all_predictions[i]
            labels += labels_batch
            predictions += preds_batch

            for y in range(len(labels_batch)):
                conf_mat_labels += labels_batch[y]
                conf_mat_preds += preds_batch[y]


        count_errors(labels, predictions, sents, epoch, domain)

        disp_labels = ["B-targ-Positive", "B-targ-Negative", "O", "I-targ-Negative", "I-targ-Positive"]
        cm = confusion_matrix(conf_mat_labels, conf_mat_preds, labels=disp_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=["B-Pos", "B-Neg", "O", "I-Neg", "I-Pos"])

        disp.plot()
        plt.savefig(f"epoch_{epoch}_{domain}_cm.png")

        with open(f"epoch_{epoch}_{domain}_results.txt", "a", encoding="utf8") as f:
            for i in range(len(labels)):  # sequence
                sent = sents[i].split()
                label_seq = labels[i]
                pred_seq = predictions[i]

                f.write("Sent:")
                for word in sent:
                    f.write(word + " ")
                f.write("\n")

                f.write("Gold: ")
                for label in label_seq:
                    f.write(label + " ")
                f.write("\n")

                f.write("Pred: ")
                for pred in pred_seq:
                    f.write(pred + " ")

                f.write("\n\n")

    return round(overall_prec / batches, 4), round(overall_rec / batches, 4), round(overall_f1 / batches, 4)
