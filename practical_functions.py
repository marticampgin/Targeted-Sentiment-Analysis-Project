from datasets import Dataset as hugging_ds
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import random


def select_domain(dataset, domains):
    domain_dataset = []
    for sample in dataset:
        if sample["domain"] in domains:
            domain_dataset.append(sample)

    return hugging_ds.from_pandas(pd.DataFrame(domain_dataset))


def few_shot_learning_datasets(dataset, train_domains, test_domains, few_shot_samples=10):
    train_domain_dataset = []
    test_domain_dataset = []
    samples_to_exclude = []
    counter = {domain: 0 for domain in test_domains}

    for sample in dataset:
        if sample["domain"] in train_domains:
            train_domain_dataset.append(sample)

        elif sample["domain"] in test_domains and counter[sample["domain"]] < few_shot_samples:
            train_domain_dataset.append(sample)
            samples_to_exclude.append(sample)
            counter[sample["domain"]] += 1

        elif (sample["domain"] in test_domains
              and counter[sample["domain"]] == few_shot_samples
              and sample not in samples_to_exclude):
            test_domain_dataset.append(sample)

    train_domain_dataset = hugging_ds.from_pandas(pd.DataFrame(train_domain_dataset))
    test_domain_dataset = hugging_ds.from_pandas(pd.DataFrame(test_domain_dataset))

    return train_domain_dataset, test_domain_dataset


def combined_train_dataset(dataset, train_domains, test_domains, seed, ratio=0.15):
    domain_samples = defaultdict(list)
    train_samples = []
    test_samples = []

    for sample in dataset:
        if sample["domain"] in train_domains:
            train_samples.append(sample)
        else:
            domain_samples[sample["domain"]].append(sample)

    random.seed(seed)
    for domain, samples in domain_samples.items():
        num_test_samples = int(ratio * len(samples))
        for _ in range(num_test_samples):
            idx = random.randint(0, len(samples) - 1)
            sample = samples.pop(idx)
            train_samples.append(sample)
        if domain in test_domains:
            test_samples += samples

    train_domain_dataset = hugging_ds.from_pandas(pd.DataFrame(train_samples))
    test_domain_dataset = hugging_ds.from_pandas(pd.DataFrame(test_samples))

    return train_domain_dataset, test_domain_dataset


def select_split_domain(dataset, domain, ratio):
    domain_dataset = []
    for sample in dataset:
        if sample["domain"] == domain:
            domain_dataset.append(sample)

    num_test_samples = int(ratio * len(domain_dataset))
    num_train_samples = len(domain_dataset) - num_test_samples

    train_dataset = domain_dataset[:num_train_samples]
    val_dataset = domain_dataset[num_train_samples:num_train_samples + num_test_samples]

    train_dataset = hugging_ds.from_pandas(pd.DataFrame(train_dataset))
    val_dataset = hugging_ds.from_pandas(pd.DataFrame(val_dataset))

    return train_dataset, val_dataset


def produce_bar_plot(dataset):
    domains = [sample["domain"] for sample in dataset]
    cnt = Counter(domains)

    domains = list(cnt.keys())
    counts = list(cnt.values())

    plt.bar(domains, counts, color="royalblue", width=0.4)
    plt.xlabel("Domains")
    plt.ylabel("Sentences")
    plt.xticks(rotation=45, horizontalalignment="center")
    plt.title("Number of sentences per domain")
    plt.show()


def produce_line_plot():
    models = ["norbert3_small", "norbert3_base", "norbert3_large", "norbert3-wiki-base"]
    domains = ["music", "literature", "screen", "games",
               "restaurants", "stage", "sports", "misc"]

    # small_res = [0.4174, 0.3857, 0.4258, 0.4416, 0.4069, 0.4884, 0.2543, 0.4815]
    # base_res = [0.4485, 0.4158, 0.4515, 0.4552, 0.4213, 0.4602, 0.2979, 0.5686]
    # large_res = [0.4327, 0.4065, 0.4311, 0.4867, 0.4336, 0.4759, 0.2725, 0.5294]
    # wiki_res = [0.3258, 0.3341, 0.3544, 0.4082, 0.3529, 0.4435, 0.1844, 0.4597]
    base_res_prod = [0.3946, 0.3789, 0.3862, 0.4836, 0.4553, 0.4024, 0.0952, 0.2750]

    # plt.plot(domains, small_res)
    # plt.plot(domains, base_res)
    # plt.plot(domains, large_res)
    # plt.plot(domains, wiki_res)
    plt.plot(domains, base_res_prod)
    # plt.legend(models)
    plt.xlabel("Domains")
    plt.xticks(rotation=45, horizontalalignment="center")
    plt.ylabel("Batista F1-scores")
    plt.title("Performance of NorBERT3-base trained on 'Products' domain")
    plt.show()


def filter_domains(preferred_domain):
    domains = ["screen", "music", "literature", "products", "games",
               "restaurants", "stage", "sports", "misc"]

    domains.remove(preferred_domain)
    return domains
