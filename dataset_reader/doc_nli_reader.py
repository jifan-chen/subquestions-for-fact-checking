from typing import Dict, Optional
import json
import logging
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer, PretrainedTransformerTokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register("doc_nli")
class DocNliReader(DatasetReader):

    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_source_length: Optional[int] = 512,
        **kwargs,
    ) -> None:
        super().__init__(manual_distributed_sharding=True, **kwargs)
        self._tokenizer = tokenizer or SpacyTokenizer()
        if isinstance(self._tokenizer, PretrainedTransformerTokenizer):
            assert not self._tokenizer._add_special_tokens
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_source_length = max_source_length

    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)

        with open(file_path, "r") as doc_nli_file:
            doc_nli_examples = json.load(doc_nli_file)
            count = 0
            for example in doc_nli_examples:
                label = "entail" if example["label"] == 'entailment' else "not_entail"
                # using the whole paragraph as premise or just the answering sent
                premise = example['premise']
                count += 1
                # if self.joint_training and count == 1000:
                #     break
                hypothesis = example["hypothesis"]
                instance = self.text_to_instance(premise,
                                                 hypothesis,
                                                 label,
                                                 )
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
        tokens = self._tokenizer.add_special_tokens(premise, hypothesis)

        if len(tokens) > self.max_source_length:
            tokens = tokens[:self.max_source_length]
        fields["tokens"] = TextField(tokens, self._token_indexers)

        metadata = {
            "premise_tokens": [x.text for x in premise],
            "hypothesis_tokens": [x.text for x in hypothesis],
        }
        fields["metadata"] = MetadataField(metadata)

        if label:
            fields["label"] = LabelField(label)

        return Instance(fields)
