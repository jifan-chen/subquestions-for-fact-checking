from typing import List, Dict
import numpy
from overrides import overrides
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import LabelField


@Predictor.register("mnli")
class MNliPredictor(Predictor):
    """
    Predictor for the [`DecomposableAttention`](../models/decomposable_attention.md) model.
    Registered as a `Predictor` with name "textual_entailment".
    """

    def predict(self, premise: str, hypothesis: str, label: str = None) -> JsonDict:
        instance = self._dataset_reader.text_to_instance(
            premise,
            hypothesis,
        )
        if instance:
            return self.predict_instance(instance)
        else:
            return None

    @overrides
    def predictions_to_labeled_instances(
            self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        # This function is used to to compute gradients of what the model predicted.
        new_instance = instance.duplicate()
        label = numpy.argmax(outputs["logits"])
        # Skip indexing, we have integer representations of the strings "entailment", etc.
        new_instance.add_field("label",
                               LabelField(int(label), skip_indexing=True))
        return [new_instance]

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        premise = json_dict["premise"]
        hypothesis = json_dict["hypothesis"]
        instance = self._dataset_reader.text_to_instance(
            premise,
            hypothesis
        )
        return instance
