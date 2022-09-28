import numpy
from overrides import overrides
from typing import List, Dict
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import LabelField


@Predictor.register("qa_nli")
class QaNliPredictor(Predictor):

    def predict(self, premise: str, hypothesis: str,
                answer_score: float = None) -> JsonDict:
        instance = self._dataset_reader.text_to_instance(
            premise,
            hypothesis,
            answer_score
        )
        if instance:
            return self.predict_instance(instance)
        else:
            return None

    def predict_batch(self, premises: List[str], hypothesises: List[str],
                      answer_scores: List[float] = None):
        instances = []
        if answer_scores is None:
            for premise, hypothesis in zip(premises, hypothesises):
                instance = self._dataset_reader.text_to_instance(
                    premise,
                    hypothesis
                )
                if instance:
                    instances.append(instance)
        else:
            for premise, hypothesis, answer_score in zip(premises,
                                                         hypothesises,
                                                         answer_scores
                                                         ):
                instance = self._dataset_reader.text_to_instance(
                    premise,
                    hypothesis,
                    answer_score=answer_score
                )
                if instance:
                    instances.append(instance)
        outputs = self.predict_batch_instance(instances)
        return outputs

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
