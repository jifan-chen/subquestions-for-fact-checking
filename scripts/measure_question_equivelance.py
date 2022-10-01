# Define QG model here
import torch
import transformers
import os
import argparse
import pandas as pd
from torch import nn
from scipy.stats import pearsonr
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_utils import (
    ModuleUtilsMixin, PushToHubMixin,
    logging
)

logger = logging.get_logger(__name__)

MODEL_URL = ""
SEP_TK = "###"
GEN_KWARGS = {
    "max_length": 128,
    "num_beams": 1,
}


class QuestionEquivalenceModel(nn.Module, ModuleUtilsMixin, PushToHubMixin):
    def __init__(self, model_name='microsoft/deberta-large'):
        super().__init__()
        # Load tokenizer and model.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       use_fast=True,
                                                       model_max_length=256)
        self.pretrain_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
        )
        self.max_length = 256
        self.config = self.pretrain_model.config
        self.data = None

    def read_from_csv_file(self, url):
        sheet_url = url.replace('/edit#gid=', '/export?format=csv&gid=')
        df = pd.read_csv(sheet_url)
        self.data = df

    def compute_pearson_correlation(self):
        recall_annot = []
        recall_pred = []
        for index, row in self.data.iterrows():
            if not isinstance(row['pred-questions'], str):
                continue
            recall_annot.append(int(row['recall-claim']) + int(row['recall-ruling']))
            recall_pred.append(int(row['recall-question-equivelance']))
        print('pearson correlation:',
              pearsonr(recall_annot, recall_pred))

    def make_prediction(self):
        with torch.no_grad():
            for index, row in self.data.iterrows():
                q1s = row['pred-questions'].split('\n')
                q2s = row['questions-all'].split('\n')
                question_paris = [(q1, q2) for q1 in q1s for q2 in q2s]
                results = self.tokenizer(
                    text=[pair[0] for pair in question_paris],
                    text_pair=[pair[1] for pair in question_paris],
                    truncation=True,
                    padding=True,
                    return_tensors='pt'
                    )
                recall = set()
                logits = self.pretrain_model(**results).logits
                for logit, pair in zip(logits, question_paris):
                    predicted_class_id = logit.argmax().item()
                    label = self.pretrain_model.config.id2label[predicted_class_id]
                    if label == 'LABEL_1':
                        recall.add(pair[1])
                print(len(recall))


def post_processing_prediction(prediction):
    questions = prediction[0].split(SEP_TK)
    return questions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_weights_from', type=str, default=None)
    parser.add_argument('--url', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()
    model = QuestionEquivalenceModel(model_name=args.model)
    model.read_from_csv_file(args.url)
    if args.load_weights_from:
        state_dict = torch.load(os.path.join(args.load_weights_from,
                                             transformers.WEIGHTS_NAME),
                                map_location="cpu")
        model.pretrain_model.load_state_dict(state_dict, strict=True)
        # release memory
        del state_dict
    # model.make_prediction()
    model.compute_pearson_correlation()
    # claim = "Joe Biden stated on August 31, 2020 in a speech: \"When I was vice" \
    #         " president, violent crime fell 15% in this country. ... The murder" \
    #         " rate now is up 26% across the nation this year " \
    #         "under Donald Trump.\" "
    # num_questions = "5"
    #
    # input_claim = "{} {} {}".format(num_questions,
    #                                 SEP_TK,
    #                                 claim)
    #
    # tokenized_claim = model.tokenizer(input_claim,
    #                                   truncation=True,
    #                                   max_length=256)
    # input_ids = torch.unsqueeze(
    #     torch.LongTensor(tokenized_claim["input_ids"]), 0
    # )
    # attention_mask = torch.unsqueeze(
    #     torch.LongTensor(tokenized_claim["attention_mask"]), 0
    # )
    # print("begin generation.........")
    # generated_tokens = model.generate(input_ids,
    #                                   attention_mask,
    #                                   do_sample=False,
    #                                   num_beams=4,
    #                                   max_length=128
    #                                   )
    # print(generated_tokens.shape)
    # raw_prediction = model.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    # print(raw_prediction)
