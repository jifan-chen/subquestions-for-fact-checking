# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset, load_metric
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBartTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from src.seq2seq_converter.utils import *

logger = logging.getLogger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "9995"  # modify if RuntimeError: Address already in use
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
# WORKER_IDS = {'A394Q9REPEHG3X', 'A8O8ZAZQL435', 'A1JOKC27MX82R'}
WORKER_IDS = {}


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task: str = field(
        default="summarization",
        metadata={
            "help": "The name of the task, should be summarization (or summarization_{dataset} for evaluating "
            "pegasus) or translation (or translation_{xx}_to_{yy})."
        },
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The path to the dataset"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    prediction_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input prediction data file to produce the predictions"},
    )

    output_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "write predictions of the model out to a path"
        }
    )

    output_format: Optional[str] = field(
        default=None,
        metadata={'help': "whether to output a csv file or the original file format"}
    )

    data_source: Optional[str] = field(
        default=None,
        metadata={'help': "whether the input data is qa-nli formatted or not"}
    )

    beam_size: Optional[int] = field(
        default=5,
        metadata={"help": "beam size during decoding"}
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    source_lang: Optional[str] = field(default=None, metadata={"help": "Source language id for translation."})
    target_lang: Optional[str] = field(default=None, metadata={"help": "Target language id for translation."})
    eval_beams: Optional[int] = field(default=None, metadata={"help": "Number of beams to use for evaluation."})
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        # else:
        #     if self.train_file is not None:
        #         extension = self.train_file.split(".")[-1]
        #         assert extension in ["csv", "json", "jsonl"], "`train_file` should be a csv or a json file."
        #     if self.validation_file is not None:
        #         extension = self.validation_file.split(".")[-1]
        #         assert extension in ["csv", "json", "jsonl"], "`validation_file` should be a csv or a json file."
        # if not self.task.startswith("decontext") and not self.task.startswith("question_convert"):
        #     raise ValueError(
        #         "`task` should be decontext or question convert"
        #     )
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        print(model_args)
        print(data_args)
        print(training_args)
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        print('Getting checkpoint...')
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files in the summarization task, this script will use the first column for the full texts and the
    # second column for the summaries (unless you specify column names for this with the `text_column` and
    # `summary_column` arguments).
    # For translation, only JSON files are supported, with one field named "translation" containing two keys for the
    # source and target languages (unless you adapt what follows).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    if data_args.train_file is not None:
        extension = data_args.train_file.split(".")[-1]
        if not extension.startswith('csv'):
            extension = 'json'
        train_dataset = load_dataset(extension,
                                     data_files=data_args.train_file,
                                     split="train")

    if data_args.validation_file is not None:
        extension = data_args.validation_file.split(".")[-1]
        if not extension.startswith('csv'):
            extension = 'json'
        eval_dataset = load_dataset(extension,
                                    data_files=data_args.validation_file,
                                    split="train")

    if data_args.prediction_file is not None:
        extension = data_args.prediction_file.split(".")[-1]
        if not extension.startswith('csv'):
            extension = 'json'
        # print(extension)
        # input()
        raw_prediction_dataset = load_dataset(
            extension,
            data_files=data_args.prediction_file,
            split="train"
        )

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # additional_tokens = ['<ans>, </ans>']
    # for i in range(150):
    #     additional_tokens.append('<extra_id_{}>'.format(i))

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, MBartTokenizer):
        model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.target_lang]
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    print('decoder start token id:', model.config.decoder_start_token_id)
    # Get the default prefix if None is passed.
    # if data_args.source_prefix is None:
    #     task_specific_params = model.config.task_specific_params
    #     if task_specific_params is not None:
    #         prefix = task_specific_params.get("prefix", "")
    #     else:
    #         prefix = ""
    # else:
    #     prefix = data_args.source_prefix

    # To serialize preprocess_function below, each of those four variables needs to be defined (even if we won't use
    # them all).
    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    def preprocess_function(examples):
        inputs = []
        targets = []
        if data_args.task.startswith("decontext"):
            if data_args.data_source == "qa-nli":
                inputs, targets = process_decontext_qanli(examples)
            else:
                inputs, targets = process_decontext_train_and_dev(examples)
        elif data_args.task.startswith("question_convert"):
            # ensure both questions and answers are not none
            if data_args.data_source == 'qa-nli':
                inputs, targets = process_question_converter_qanli(examples)
            else:
                inputs, targets = process_question_converter_train_and_dev(examples)
        elif data_args.task.startswith("question_generate"):
            inputs, targets = process_question_generator_train_and_dev(examples)
        elif data_args.task.startswith("all_question_generate_ruling"):
            inputs, targets = process_qg_ruling_train_and_dev(
                examples,
                training=training_args.do_train,
                sampling=False,
                worker_ids=WORKER_IDS)
        elif data_args.task.startswith("one_question_generate_ruling"):
            inputs, targets = process_qg_ruling_train_and_dev(
                examples,
                training=training_args.do_train,
                sampling=True,
                worker_ids=WORKER_IDS)
        elif data_args.task.startswith("all_question_generate"):
            inputs, targets = process_qg_train_and_dev(
                examples,
                training=training_args.do_train,
                sampling=False,
                worker_ids=WORKER_IDS)
        elif data_args.task.startswith("one_question_generate"):
            inputs, targets = process_qg_train_and_dev(
                examples,
                training=training_args.do_train,
                sampling=True,
                worker_ids=WORKER_IDS)
        else:
            if data_args.data_source == 'qa-nli':
                inputs, targets = process_esnli_qanli(examples)
            else:
                inputs, targets = process_esnli_train_and_dev(examples)
        
        # for ipt, tgt in zip(inputs, targets):
        #     print(ipt)
        #     print(tgt)

        model_inputs = tokenizer(text=inputs,
                                 max_length=data_args.max_source_length,
                                 padding=padding,
                                 truncation=True,
                                 add_special_tokens=True
                                 )

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets,
                               max_length=max_target_length,
                               padding=padding,
                               truncation=True
                               )
            # print(labels)
            # decoded_labels = tokenizer.batch_decode(labels.input_ids,
            #                                         skip_special_tokens=False)
            # print(decoded_labels)
            # input()
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        # train_dataset = datasets["train"]
        column_names = train_dataset.column_names
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        print("training data features:", train_dataset)
        print("training examples:", len(train_dataset))
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        column_names = eval_dataset.column_names
        # eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        column_names = raw_prediction_dataset.column_names
        if data_args.max_val_samples is not None:
            prediction_dataset = raw_prediction_dataset.select(
                range(data_args.max_val_samples)
            )
        else:
            prediction_dataset = raw_prediction_dataset

        prediction_dataset = prediction_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache
        )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # Metric
    metric_name = "bleu" if data_args.task.startswith("question_convert") else "rouge"
    metric = load_metric(metric_name)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        if metric_name == "bleu":
            decoded_labels = [[label.split()] for label in decoded_labels]
            decoded_preds = [pred.split() for pred in decoded_preds]

        # for pred, lable in zip(decoded_preds, decoded_labels):
        #     print(pred)
        #     print(lable)
        #     input()

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        # print(result)
        # Extract a few results from ROUGE
        if metric_name == "rouge":
            result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        else:
            result = {"bleu": result["bleu"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)

        return result

    # Initialize our Trainer
    if training_args.do_train:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        )
    elif training_args.do_predict:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=None,
            eval_dataset=None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=None
        )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            model_path = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            model_path = model_args.model_name_or_path
        else:
            model_path = None
        train_result = trainer.train(model_path=model_path)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))
    
    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        results = trainer.evaluate(num_beams=data_args.beam_size,
                                   max_length=data_args.max_target_length
                                   )
        # predictions = trainer.predict()
        output_eval_file = os.path.join(training_args.output_dir, "eval_results_seq2seq.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    if training_args.do_predict:
        logger.info("*** Begin prediction ***")
        if data_args.task.startswith("one_question_generate_ruling") or data_args.task.startswith("one_question_generate"):
            decoded_preds = []
            for i in range(len(prediction_dataset)):
                # print(prediction_dataset[i])
                # print(tokenizer.decode(prediction_dataset[i]['input_ids'][3:]))
                # input()
                # when number of questions to generate is part of claim, right now hardcoded to 3
                num_gen = int(tokenizer.decode(prediction_dataset[i]['input_ids']).split(SEP_TK)[0])
                p = model.generate(
                    input_ids=torch.tensor([prediction_dataset[i]['input_ids'][3:]],
                    dtype=torch.int32, device=torch.device('cuda:0')),
                    do_sample=True,
                    top_k=80,
                    top_p=0.95,
                    max_length=100,
                    num_return_sequences=num_gen)
                dp = tokenizer.batch_decode(p, skip_special_tokens=True)
                decoded_preds.append(SEP_TK.join(dp))
            # decoded_preds = set(decoded_preds)
        else:
            results = trainer.predict(test_dataset=prediction_dataset,
                                      num_beams=data_args.beam_size,
                                      max_length=data_args.max_target_length)
            predictions, label_ids, metrics = results[0], results[1], results[2]
            decoded_preds = tokenizer.batch_decode(predictions,
                                               skip_special_tokens=True)
        if data_args.task.startswith('decontext'):
            write_decontext_predictions_out(raw_prediction_dataset,
                                            decoded_preds,
                                            data_args.output_path,
                                            output_format=data_args.output_format,
                                            data_source=data_args.data_source)

        if data_args.task.startswith('question_convert'):
            write_question_converter_predictions_out(
                raw_prediction_dataset,
                decoded_preds,
                data_args.output_path,
                output_format=data_args.output_format,
                data_source=data_args.data_source
            )

        if data_args.task.startswith("esnli"):
            write_esnli_predictions_out(
                raw_prediction_dataset,
                decoded_preds,
                data_args.output_path,
                output_format=data_args.output_format,
                data_source=data_args.data_source
            )
        
        if data_args.task.startswith("question_generate"):
            write_question_generator_predictions_out(
                raw_prediction_dataset,
                decoded_preds,
                data_args.output_path,
                output_format=data_args.output_format,
                data_source=data_args.data_source
            )
        if data_args.task == "all_question_generate_ruling":
            write_all_question_generator_ruling_predictions_out(
                raw_prediction_dataset,
                decoded_preds,
                data_args.output_path,
                output_format=data_args.output_format,
                data_source=data_args.data_source
            )
        if data_args.task == "one_question_generate_ruling":
            write_one_question_generator_ruling_predictions_out(
                raw_prediction_dataset,
                decoded_preds,
                data_args.output_path,
                output_format=data_args.output_format,
                data_source=data_args.data_source
            )
        if data_args.task == "all_question_generate":
            write_all_question_generator_predictions_out(
                raw_prediction_dataset,
                decoded_preds,
                data_args.output_path,
                output_format=data_args.output_format,
                data_source=data_args.data_source
            )
        if data_args.task == "one_question_generate":
            write_one_question_generator_predictions_out(
                raw_prediction_dataset,
                decoded_preds,
                data_args.output_path,
                output_format=data_args.output_format,
                data_source=data_args.data_source
            )
    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
