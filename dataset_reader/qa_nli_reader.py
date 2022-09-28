import itertools
from typing import Dict, Optional
import json
import logging
import numpy as np
import random
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer, PretrainedTransformerTokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register("qa_nli")
class QaNliReader(DatasetReader):

    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        combine_input_fields: Optional[bool] = None,
        use_full_context: Optional[bool] = False,
        use_decontext: Optional[bool] = True,
        joint_training: Optional[bool] = False,
        joint_eval: Optional[bool] = False,
        use_answer_score: Optional[bool] = False,
        max_source_length: Optional[int] = 512,
        mnli_path: str = None,
        qa_example_ratio: float = 1,
        **kwargs,
    ) -> None:
        super().__init__(manual_distributed_sharding=True, **kwargs)
        self._tokenizer = tokenizer or SpacyTokenizer()
        if isinstance(self._tokenizer, PretrainedTransformerTokenizer):
            assert not self._tokenizer._add_special_tokens
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if combine_input_fields is not None:
            self._combine_input_fields = combine_input_fields
        else:
            self._combine_input_fields = isinstance(self._tokenizer, PretrainedTransformerTokenizer)
        self.use_full_context = use_full_context
        self.use_decontext = use_decontext
        self.joint_training = joint_training
        self.joint_eval = joint_eval
        self.use_answer_scores = use_answer_score
        self.max_source_length = max_source_length
        self.mnli_path = mnli_path
        self.qa_example_ratio = qa_example_ratio

    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)

        with open(file_path, "r") as qed_nli_file:
            qa_nli_examples = [json.loads(line) for line in qed_nli_file]
            num_qa_nli_examples = len(qa_nli_examples)
            mnli_examples = []
            if self.mnli_path:
                with open(self.mnli_path) as fin:
                    for line in fin:
                        mnli_examples.append(json.loads(line))

            if 'train' in file_path:
                if self.joint_training:
                    all_examples = mnli_examples[: int(1 * num_qa_nli_examples)] + \
                                   qa_nli_examples[: int(self.qa_example_ratio * num_qa_nli_examples)]
                else:
                    all_examples = qa_nli_examples
                random.shuffle(all_examples)
            else:
                if self.joint_eval:
                    all_examples = mnli_examples + \
                                   qa_nli_examples[: int(self.qa_example_ratio * num_qa_nli_examples)]
                else:
                    all_examples = qa_nli_examples

            count = 0
            for example in all_examples:
                label = "entail" if example["is_correct"] else "not_entail"
                # using the whole paragraph as premise or just the answering sent
                if self.use_full_context:
                    premise = example["paragraph_text"]
                else:
                    if self.use_decontext:
                        # premise = example['decontext_answer_sent_text']
                        premise = example['decontext_answer_sent_text']
                    else:
                        premise = example['answer_sent_text']
                count += 1
                # if self.joint_training and count == 1000:
                #     break
                hypothesis = example["question_statement_text"]
                if self.use_answer_scores:
                    if 'answer_score' in example.keys():
                        answer_score = example['answer_score']
                    else:
                        # raise ValueError('answer score not found')
                        answer_score = 0
                else:
                    answer_score = None
                # assert answer_score is not None
                instance = self.text_to_instance(premise,
                                                 hypothesis,
                                                 label,
                                                 answer_score)
                if instance:
                    yield instance

    @overrides
    def text_to_instance(
        self,  # type: ignore
        premise: str,
        hypothesis: str,
        label: str = None,
        answer_score: float = None
    ) -> Instance:

        fields: Dict[str, Field] = {}
        premise = self._tokenizer.tokenize(premise)
        hypothesis = self._tokenizer.tokenize(hypothesis)
        if answer_score is not None:
            fields['answer_scores'] = ArrayField(np.array(answer_score))
        if self._combine_input_fields:
            tokens = self._tokenizer.add_special_tokens(premise, hypothesis)
            if len(tokens) > self.max_source_length:
                tokens = tokens[:self.max_source_length]
            fields["tokens"] = TextField(tokens, self._token_indexers)
        else:
            premise_tokens = self._tokenizer.add_special_tokens(premise)
            hypothesis_tokens = self._tokenizer.add_special_tokens(hypothesis)
            fields["premise"] = TextField(premise_tokens, self._token_indexers)
            fields["hypothesis"] = TextField(hypothesis_tokens, self._token_indexers)

            metadata = {
                "premise_tokens": [x.text for x in premise_tokens],
                "hypothesis_tokens": [x.text for x in hypothesis_tokens],
            }
            fields["metadata"] = MetadataField(metadata)
        # if len(fields["tokens"]) > 512:
        #     fields["tokens"] = fields["tokens"][:512]

        if label:
            fields["label"] = LabelField(label)

        return Instance(fields)