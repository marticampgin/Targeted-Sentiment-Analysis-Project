# Targeted-Sentiment-Analysis-Project

This project was an exam at the University of Oslo in the course IN5550 - Neural Methods in NLP. The goal was to fine-tune a Norwegian BERT-model for the task Targeted Sentiment Analysis for Norwegian product reviews across different categories.
My research comprised exploring cross-domain effects, i.e. training models on data from different domains, performing error analysis, and seeing how the model behaves in general. All the methodology, results, model, as well as the dataset, are described in provided the PDF paper. 

Here is a brief description of the associated .py files and their purpose:

- [bert_train.py](bert_train.py): the main train script, including training and eval loops.
- [configuration_norbert.py](configuration_norbert.py): configuration file including all the hyperparameters for the NorbertModel.
- [evaluate_tsa.py](evaluate_tsa.py) / [ner_eval.py](ner_eval.py): files including the code to perform correct evaluation for the task of Targeted Sentiment Analysis.
- [modeling_norbert.py](modeling_norbert.py): code of the Transformer architecture (more specifically, the encoder), as well as Huggingface wrappers for different tasks.
- [practical_functions.py](practical_functions.py): file with various functions used to conduct different experiments.
- [read_dataset.py](read_dataset.py): code that reads and uploads the dataset.

The dataset utilized in the experiments was specifically tailored for this task, and its tailored version has been unfortunately lost, but can be recreated following the links provided in the PDF paper. 
