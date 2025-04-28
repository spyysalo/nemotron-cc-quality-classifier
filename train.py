import sys

import numpy as np
import evaluate

from argparse import ArgumentParser

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import classification_report, confusion_matrix


DEFAULT_MODEL = 'intfloat/multilingual-e5-large-instruct'

DEFAULT_DATASET = 'spyysalo/nemotron-cc-10K-sample'


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--dataset', default=DEFAULT_DATASET)
    ap.add_argument('--model', default=DEFAULT_MODEL)
    return ap


def compute_metrics(pred_labels):
    metrics = {
        m: evaluate.load(m)
        for m in ['accuracy', 'precision', 'recall', 'f1']
    }

    preds, labels = pred_labels
    preds = np.argmax(preds, axis=1)

    results = {}
    for n, m in metrics.items():
        kwargs = { 'predictions': preds, 'references': labels }
        if n == 'accuracy':
            results[n] = m.compute(**kwargs)[n]
        else:
            for a in ('micro', 'macro'):
                results[f'{a}_{n}'] = m.compute(**kwargs, average=a)[n]

    print('Classification report:')
    print(classification_report(labels, preds))
    print('Confusion matrix')
    print(confusion_matrix(labels, preds))

    return results


def freeze_base_model(model):
    for param in model.base_model.parameters():
        param.requires_grad = False


def main(argv):
    args = argparser().parse_args(argv[1:])
    
    dataset = load_dataset(args.dataset)
    dataset = dataset['train'].train_test_split(test_size=1000)
    dataset = dataset.filter(lambda e: e['label'] in (0, 4))

    def binarize(e):
        e['label'] = 0 if e['label'] < 2 else 1
        return e

    dataset = dataset.map(binarize)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
    )
    freeze_base_model(model)
    
    tokenize = lambda e: tokenizer(e['text'], truncation=True)
    dataset = dataset.map(tokenize, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_args = TrainingArguments(
        output_dir='output',
        learning_rate=3e-3,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=1,
        weight_decay=0.01,
        eval_strategy='steps',
        save_strategy='steps',
        eval_steps=250,
        save_steps=250,
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print(trainer.evaluate())


if __name__ == '__main__':
    sys.exit(main(sys.argv))
