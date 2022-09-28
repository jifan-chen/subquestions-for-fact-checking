import argparse
from allennlp.common.util import import_module_and_submodules
from allennlp.predictors.predictor import Predictor


def main(args):
    if args.model == "nq-nli":
        model_path = "./nli_models/nq-nli.tar.gz"
        model_name = "qa_nli"
    elif args.model == 'mnli':
        model_path = "./nli_models/mnli.tar.gz"
        model_name = "textual_entailment"
    elif args.model == 'doc-nli':
        model_path = "./nli_models/doc-nli.tar.gz"
        model_name = "qa_nli"
    else:
        raise ValueError('no model named {}'.format(args.model))

    predictor = Predictor.from_path(
        model_path,
        model_name,
        cuda_device=0)

    premise = "The jobless rate for Hispanics hit a record low of 3.9% in" \
              " September, while African Americans maintained its lowest rate " \
              "ever, 5.5%."
    hypothesis = "The unemployment rate of African Americans is historically low."

    results = predictor.predict(
        premise=premise,
        hypothesis=hypothesis
    )
    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='doc_nli')
    parsed_args = parser.parse_args()
    import_module_and_submodules("predictors")
    import_module_and_submodules("dataset_reader")
    import_module_and_submodules("model")
    main(parsed_args)
