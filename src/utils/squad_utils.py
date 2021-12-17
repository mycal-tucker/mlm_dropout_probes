# Taken from https://github.com/huggingface/notebooks/blob/master/examples/question_answering.ipynb
import collections

import numpy as np
import torch
import transformers
from datasets import load_dataset
from datasets import load_metric
from scipy.special import softmax
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from tqdm.auto import tqdm
from transformers import AdamW
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import default_data_collator


def load_parts(model_checkpoint="distilbert-base-uncased", datasets=None, num_train=1000, num_val=100, shuffle=True):
    if datasets is None:
        datasets = load_dataset("squad_v2")
    elif isinstance(datasets, str):
        datasets = load_dataset('json', data_files={'train': datasets, 'validation': datasets}, field='data')
    if shuffle:
        datasets['train'] = datasets['train'].shuffle()
        datasets['validation'] = datasets['validation'].shuffle()
    datasets['train'] = datasets['train'].select([i for i in range(min(num_train, len(datasets['train'])))])
    datasets['validation'] = datasets['validation'].select([i for i in range(min(num_val, len(datasets['validation'])))])
    batch_size = 8  # Was 8
    max_length = 384  # The maximum length of a feature (question and context)
    doc_stride = 128  # The authorized overlap between two part of the context when splitting it is needed.

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

    for i, example in enumerate(datasets["train"]):
        print("example", example)
        if len(tokenizer(example["question"], example["context"])["input_ids"]) > 384:
            break
    example = datasets["train"][i]
    tokenized_example = tokenizer(
        example["question"],
        example["context"],
        max_length=max_length,
        truncation="only_second",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        stride=doc_stride
    )
    sequence_ids = tokenized_example.sequence_ids()
    answers = example["answers"]
    custom_corpus = isinstance(answers, list)
    if custom_corpus:
        answers = answers[0]
    start_char = answers["answer_start"]
    if isinstance(start_char, list):
        start_char = start_char[0]
    end_char = start_char + len(answers["text"][0])

    # Start token index of the current span in the text.
    token_start_index = 0
    while sequence_ids[token_start_index] != 1:
        token_start_index += 1

    # End token index of the current span in the text.
    token_end_index = len(tokenized_example["input_ids"][0]) - 1
    while sequence_ids[token_end_index] != 1:
        token_end_index -= 1

    # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
    offsets = tokenized_example["offset_mapping"][0]
    if offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char:
        # Move the token_start_index and token_end_index to the two ends of the answer.
        # Note: we could go after the last offset if the answer is the last word (edge case).
        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
            token_start_index += 1
        start_position = token_start_index - 1
        while offsets[token_end_index][1] >= end_char:
            token_end_index -= 1
        end_position = token_end_index + 1
        print(start_position, end_position)
    else:
        print("The answer is not in this feature.")

    pad_on_right = tokenizer.padding_side == "right"

    def prepare_train_features(examples):
        # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            if custom_corpus:
                answers = answers[0]
            # If no answers are given, set the cls_index as answer.
            if not custom_corpus and len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"]
                if isinstance(start_char, list):
                    start_char = start_char[0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    tokenized_datasets = datasets.map(prepare_train_features, batched=True, remove_columns=datasets["train"].column_names)
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    args = TrainingArguments(
        f"test-squad",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=1,
        weight_decay=0.01,
    )

    data_collator = default_data_collator
    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    return datasets, trainer, tokenizer, pad_on_right, max_length, doc_stride


def train_tail_layers(model, train_dataset, num_epochs=3):
    # setup GPU/CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # move model over to detected device
    model.to(device)
    # activate training mode of model
    model.train()
    # initialize adam optimizer with weight decay (reduces chance of overfitting)
    optim = AdamW(model.parameters(), lr=5e-5)

    max_len = max([elt[0].shape[0] for elt in train_dataset])
    # initialize data loader for training data
    # Need to pad to create a non-ragged tensor
    inps = torch.tensor([np.pad(elt[0], [(0, max_len - elt[0].shape[0]), (0, 0)]) for elt in train_dataset])
    starts = torch.tensor([elt[1] for elt in train_dataset])
    ends = torch.tensor([elt[2] for elt in train_dataset])
    tensor_dataset = TensorDataset(inps, starts, ends)
    train_loader = DataLoader(tensor_dataset, batch_size=16)

    for epoch in range(num_epochs):
        # set model to train mode
        model.train()
        # setup loop (we use tqdm for the progress bar)
        loop = tqdm(train_loader, leave=True)
        for batch_idx, batch in enumerate(loop):
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all the tensor batches required for training
            input_ids = batch[0].to(device)
            start_positions = batch[1].to(device)
            end_positions = batch[2].to(device)
            # train model on batch and return outputs (incl. loss)
            outputs = model(torch.tensor(input_ids, dtype=torch.float32).to(device))
            start_logits, end_logits = outputs
            loss_fn = nn.CrossEntropyLoss()
            start_loss = loss_fn(start_logits, start_positions)
            end_loss = loss_fn(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optim.step()
            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())


def postprocess_qa_predictions(examples, features, raw_predictions, tokenizer, n_best_size=20, max_answer_length=30,
                               id_field_name="id", context_field_name='context'):
    import collections

    example_id_to_index = {k: i for i, k in enumerate(examples[id_field_name])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)
    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples[id_field_name])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Logging.
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None  # Only used if squad_v2 is True.
        valid_answers = []

        context = example[context_field_name]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}

        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        answer = best_answer["text"] if min_null_score is None or best_answer["score"] > min_null_score else ""
        predictions[example[id_field_name]] = answer

    return predictions


def eval_qa(datasets, trainer, tokenizer, pad_on_right, max_length, doc_stride, logits=None):
    for batch in trainer.get_eval_dataloader():
        break
    # Probably useless and left over from prior demo.
    if not logits:
        batch = {k: v.to(trainer.args.device) for k, v in batch.items()}
        with torch.no_grad():
            output = trainer.model(**batch)

    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # We keep the example_id that gave us this feature and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    validation_features = datasets["validation"].map(
        prepare_validation_features,
        batched=True,
        remove_columns=datasets["validation"].column_names
    )

    raw_predictions = trainer.predict(validation_features)
    logit_predictions = raw_predictions.predictions
    if logits is not None:
        print("Replacing predictions")
        logit_predictions = logits
    validation_features.set_format(type=validation_features.format["type"], columns=list(validation_features.features.keys()))
    examples = datasets["validation"]
    features = validation_features

    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    overall_start_token_logits = softmax(logit_predictions[0], axis=1)
    overall_end_token_logits = softmax(logit_predictions[1], axis=1)

    final_predictions = postprocess_qa_predictions(datasets["validation"], validation_features,
                                                   logit_predictions, tokenizer)
    # print("Final predictions", final_predictions)
    # print("Raw predictions", raw_predictions)
    answer_start_likelihoods = []
    for idx, prediction in enumerate(final_predictions.values()):
        pred_text = prediction
        answers = datasets["validation"][idx]["answers"]
        if isinstance(answers, list):
            answers = answers[0]
        # print("Context:\t", datasets["validation"][idx]["context"])
        # print("True text:\t\t", answers["text"])
        # print("Predicted text:\t", pred_text)
        answer_start_char_idx = answers['answer_start']
        if isinstance(answer_start_char_idx, list):  # Real squad dataset has multiple possible answers
            continue
        offset_mapping = validation_features[idx]['offset_mapping']
        for token_idx, pred in enumerate(overall_start_token_logits[idx]):
            matching_chars = offset_mapping[token_idx]
            if matching_chars is None:
                continue
            if matching_chars[0] <= answer_start_char_idx < matching_chars[1]:
                # print("Predicted right start with likelihood", pred)
                next_token_pred = overall_start_token_logits[idx][token_idx + 1]
                answer_start_likelihoods.append(pred + next_token_pred)
                break
        # print()
    print("Correct start predictions", answer_start_likelihoods)
    even_idx_preds = np.mean([answer_start_likelihoods[i * 2] for i in range(int(len(answer_start_likelihoods) / 2))])
    odd_idx_preds = np.mean([answer_start_likelihoods[i * 2 + 1] for i in range(int(len(answer_start_likelihoods) / 2))])
    print("Mean even start preds", even_idx_preds)
    print("Mean odd start preds", odd_idx_preds)
    print("Mean likelihood for right start", np.mean(answer_start_likelihoods))
    metric = load_metric("squad_v2")

    formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in
                                 final_predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]
    result = metric.compute(predictions=formatted_predictions, references=references)
    print("Result", result)
    return result
