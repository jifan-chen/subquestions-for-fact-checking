import argparse
import json
import pandas as pd
import numpy as np
from typing import List
from collections import Counter, defaultdict
from src.model.polar_question_converter import PolarQuestionConverter
from src.scripts.convert_polar_q import convert_polar_questions_by_gpt3
from allennlp.common.util import import_module_and_submodules
from sklearn.metrics import f1_score, precision_score, recall_score
from allennlp.predictors.predictor import Predictor
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import spacy
nlp = spacy.load("en_core_web_sm")


def compute_majority_vote(evd_votes):
    res = []
    for i in range(len(evd_votes[0])):
        vote_one_question = []
        for j in range(len(evd_votes[0][0])):
            counter = Counter()
            for k in range(len(evd_votes)):
                vote = [evd_votes[k][i][j]]
                counter.update(vote)
            majority_vote = counter.most_common(1)
            vote_one_question.append(majority_vote[0][0])
        res.append(vote_one_question)
    return res


def aggregate_evd_annotations(evidence: List[List]):
    evidence = np.array(evidence)
    evidence = (np.sum(evidence, axis=0) > 0).astype(int)
    return evidence


def read_annotations(annotations_df, using_prediction=False):
    id2annotations = {}
    for i, row in annotations_df.iterrows():
        # sub_claims = row['Input.sub-claim'].split('[SEP]')
        # sub_claims.append(row['Input.claim'])
        evds = row['Input.full_evidence'].split('\n')
        evds = [evd for evd in evds if len(evd)]
        claim = row['Input.claim']
        claim_id = row['Input.id']
        evd_masks = []
        if using_prediction:
            questions = row['Input.predicted-questions'].split('[SEP]')
            statements = row['Input.predicted-statements'].split('\n')
            negations = row['Input.predicted-statements-negate'].split('\n')
        else:
            questions = row['Input.consensus-question'].split('[SEP]')
            statements = row['Input.statements'].split('\n')
            negations = row['Input.statements-negate'].split('\n')
        questions = [q for q in questions if len(q)]
        if claim_id not in id2annotations.keys():
            evidence_annotation = []

        for idx, qd in enumerate(json.loads(row['Answer.user-output'])):
            evd_mask = [
                2 if item == 'refute' else (1 if item == 'support' else 0)
                for item in qd['choices']]
            assert len(evd_mask) == len(evds), \
                "length of evidence mask {} not equal to the length " \
                "of the full evidence {}".format(len(evd_mask), len(evds))
            evd_masks.append(evd_mask)

        evidence_annotation.append(evd_masks)
        id2annotations[claim_id] = {
            "claim": claim,
            "evidence": evds,
            "who": row['Input.who'],
            "when_where": row['Input.when_where'],
            "questions": questions,
            'evidence_annotation': evidence_annotation,
            'statements': statements,
            'negations': negations
        }

    return id2annotations


def get_random_baseline_performance(annotations):
    all_gold_annotations = []
    for example_id, annot in annotations.items():
        evd_annotations = annot['evidence_annotation']
        gold_annotations = compute_majority_vote(evd_annotations)
        gold_annotations = aggregate_evd_annotations(gold_annotations)
        all_gold_annotations.extend(gold_annotations)
    counter = Counter(all_gold_annotations)
    positive_threshold = counter[1] / (counter[0] + counter[1])
    random_predictions = []
    for i in range(len(all_gold_annotations)):
        threshold = np.random.uniform(0, 1, 1)
        if threshold <= positive_threshold:
            random_predictions.append(1)
        else:
            random_predictions.append(0)
    f1 = f1_score(all_gold_annotations, random_predictions)
    p = precision_score(all_gold_annotations, random_predictions)
    r = recall_score(all_gold_annotations, random_predictions)
    return f1, p, r


def get_human_performance(annotations):
    f1s = []
    precisions = []
    recalls = []
    for example_id, annot in annotations.items():
        evd_annotations = annot['evidence_annotation']
        f1 = 0
        precision = 0
        recall = 0
        for i in range(len(evd_annotations)):
            gold_annotations = []
            for j in range(len(evd_annotations)):
                if j != i:
                    gold_annotations.append(evd_annotations[j])
            gold_annotations = compute_majority_vote(gold_annotations)
            gold_annotations = aggregate_evd_annotations(gold_annotations)
            prediction = aggregate_evd_annotations(evd_annotations[i])
            f1 += f1_score(gold_annotations, prediction)
            precision += precision_score(gold_annotations, prediction)
            recall += recall_score(gold_annotations, prediction)
        f1s.append(f1 / len(evd_annotations))
        precisions.append(precision / len(evd_annotations))
        recalls.append(recall / len(evd_annotations))
    return float(np.mean(f1s)), float(np.mean(precisions)), float(np.mean(recalls))


def get_bm25_performance(annotations, num_to_retrieve=None):
    f1_q = []
    f1_c = []
    for example_id, annot in annotations.items():
        questions = annot['questions']
        # sub_claims = [converter.convert(q)[0] for q in annot['questions']]
        claim = annot['claim']
        evidence = annot['evidence'][1:]
        golds = compute_majority_vote(annot['evidence_annotation'])
        golds = aggregate_evd_annotations(golds)[1:]
        num_gold = np.sum(golds) if not num_to_retrieve else num_to_retrieve
        if np.sum(golds) == 0:
            continue
        tokenized_corpus = [evd.split(" ") for evd in evidence]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = []
        for q in questions:
            score = bm25.get_scores([w.text for w in nlp(q)])
            scores.append(score)
        scores = np.array(scores)
        index_question = np.argsort(-scores.flatten()) % len(evidence)
        top_k_index_question = set()
        # selected_scores = []
        for ids, i in enumerate(index_question):
            top_k_index_question.add(i)
            if len(top_k_index_question) == num_gold:
                break
        # print(top_k_index_question)
        # merged_top_k_index = Counter()
        # for ids in top_k_index_question:
        #     merged_top_k_index.update(ids)
        # top_k_index_question = merged_top_k_index.most_common(top_k)
        # top_k_index_question = [e[0] for e in top_k_index_question]
        score = bm25.get_scores(
            [w.text for w in nlp(claim)])
        top_k_index_claim = np.argsort(-score)[:num_gold]
        predicts_question = np.zeros(len(evidence), dtype=int)
        predicts_claim = np.zeros(len(evidence), dtype=int)

        for idx in top_k_index_question:
            predicts_question[idx] = 1
        for idx in top_k_index_claim:
            predicts_claim[idx] = 1

        f1_q.append(f1_score(golds, predicts_question))
        f1_c.append(f1_score(golds, predicts_claim))

    return np.mean(f1_q), np.mean(f1_c)


def get_nli_performance_merged(annotations,
                               sorted_indices1,
                               sorted_indices2,
                               num_to_retrieve=None):
    f1s = []
    precisions = []
    recalls = []
    for example_id, annot in annotations.items():
        evidence = annot['evidence'][1:]
        golds = compute_majority_vote(annot['evidence_annotation'])
        golds = aggregate_evd_annotations(golds)[1:]
        num_gold = np.sum(golds) if not num_to_retrieve else num_to_retrieve
        if np.sum(golds) == 0:
            continue
        indices1 = sorted_indices1[example_id]['indices']
        scores1 = sorted_indices1[example_id]['scores']
        # print(indices1)
        # print(scores1)
        indices2 = sorted_indices2[example_id]['indices']
        scores2 = sorted_indices2[example_id]['scores']
        # print(indices2)
        # print(scores2)
        predicts = np.zeros(len(evidence), dtype=int)
        top_k_index_question1 = []
        top_k_scores1 = []
        top_k_index_question2 = []
        top_k_scores2 = []
        for ids, i in enumerate(indices1):
            if i not in top_k_index_question1:
                top_k_index_question1.append(i)
                top_k_scores1.append(scores1[ids])
            if len(top_k_index_question1) == num_gold:
                break

        for ids, i in enumerate(indices2):
            if i not in top_k_index_question2:
                top_k_index_question2.append(i)
                top_k_scores2.append(scores2[ids])
            if len(top_k_index_question2) == num_gold:
                break

        combined_index = [(i, s) for i, s in zip(top_k_index_question1, top_k_scores1)]
        combined_index.extend([(i, s) for i, s in zip(top_k_index_question2, top_k_scores2)])
        combined_index.sort(key=lambda x: x[1], reverse=True)
        # print(top_k_index_question1)
        # print(top_k_scores1)
        # print(top_k_index_question2)
        # print(top_k_scores2)
        # print(combined_index)
        top_k_index_question = []
        for idx_score in combined_index:
            idx = idx_score[0]
            if idx not in top_k_index_question:
                top_k_index_question.append(idx)
            if len(top_k_index_question) == num_gold:
                break

        for idx in top_k_index_question:
            predicts[idx] = 1
        # print(golds)
        # print(predicts)
        # print(f1_score(golds, predicts))
        # input()

        # all_golds.extend(golds)
        # all_predicts.extend(predicts)
        f1s.append(f1_score(golds, predicts))
        precisions.append(precision_score(golds, predicts))
        recalls.append(recall_score(golds, predicts))
    # f1 = f1_score(all_golds, all_predicts)
    # p = precision_score(all_golds, all_predicts)
    # r = recall_score(all_golds, all_predicts)
    return float(np.mean(f1s)), float(np.mean(precisions)), float(
        np.mean(recalls))


def get_nli_performance(annotations,
                        sorted_indices,
                        merged=False,
                        num_to_retrieve=None):
    f1s = []
    precisions = []
    recalls = []
    for example_id, annot in annotations.items():
        evidence = annot['evidence'][1:]
        golds = compute_majority_vote(annot['evidence_annotation'])
        golds = aggregate_evd_annotations(golds)[1:]
        num_gold = np.sum(golds) if not num_to_retrieve else num_to_retrieve
        if np.sum(golds) == 0:
            continue
        indices = sorted_indices[example_id]['indices']
        scores = sorted_indices[example_id]['scores']
        predicts = np.zeros(len(evidence), dtype=int)
        top_k_index_question = []
        top_k_scores = []

        for ids, i in enumerate(indices):
            if i not in top_k_index_question:
                top_k_index_question.append(i)
                top_k_scores.append(scores[ids])
            if len(top_k_index_question) == num_gold:
                break
        for idx in top_k_index_question:
            predicts[idx] = 1
        # all_golds.extend(golds)
        # all_predicts.extend(predicts)
        f1s.append(f1_score(golds, predicts))
        precisions.append(precision_score(golds, predicts))
        recalls.append(recall_score(golds, predicts))
    # f1 = f1_score(all_golds, all_predicts)
    # p = precision_score(all_golds, all_predicts)
    # r = recall_score(all_golds, all_predicts)
    return float(np.mean(f1s)), float(np.mean(precisions)), float(np.mean(recalls))


def get_nli_scores_claim(annotations,
                         predictor):
    res = defaultdict(dict)
    for example_id, annot in tqdm(annotations.items()):
        claim = annot['claim']
        evidence = annot['evidence'][1:]
        score = []
        for evd in evidence:
            results = predictor.predict(
                premise=evd,
                hypothesis=claim
            )
            # score.append(max(results['probs'][0], results['probs'][1]))
            score.append(results['probs'][0])
        score = np.array(score)
        res[example_id]['scores'] = -np.sort(-score)
        res[example_id]['indices'] = np.argsort(-score)
    return res


def get_nli_scores_question(annotations,
                            predictor,
                            question_converter,
                            do_negation=False,
                            top_k=3):

    res = defaultdict(dict)
    tmp_output = []
    for example_id, annot in tqdm(annotations.items()):
        questions = annot['questions']
        statements = []
        negates = []
        # for q in questions:
        #     raw_res = convert_polar_questions_by_gpt3(q)
        #     raw_res = raw_res['choices'][0]['text']
        #     statements.append(raw_res.split('|')[0].strip())
        #     negates.append(raw_res.split('|')[-1].strip())
        if do_negation:
            # sub_claims = [question_converter.convert(q)[1] for q in questions]
            # sub_claims = negates
            sub_claims = annot['negations']
        else:
            # sub_claims = [question_converter.convert(q)[0] for q in questions]
            # sub_claims = statements
            sub_claims = annot['statements']
        # for j in range(3):
        #     tmp_output.append({'statements': '\n'.join(statements),
        #                        'statements-negate': '\n'.join(negates)})
        evidence = annot['evidence'][1:]
        scores = []
        for sub_claim in sub_claims:
            score = []
            for evd in evidence:
                results = predictor.predict(
                    premise=evd,
                    hypothesis=sub_claim
                )
                score.append(results['probs'][0])
            scores.append(score)

        scores = np.array(scores)
        index_question = np.argsort(-scores.flatten()) % len(evidence)
        scores = -np.sort(-scores.flatten())
        res[example_id]['scores'] = scores
        res[example_id]['indices'] = index_question

    # output_frame = pd.DataFrame(tmp_output)
    # print(output_frame)
    # output_frame.to_csv('./output_files/question-predicated-statement-retrieval.csv', index=False)
    return res


def merge_nli_scores(scores1, scores2):
    res = defaultdict(dict)
    for example_id in scores1.keys():
        indices1 = scores1[example_id]['indices']
        indices2 = scores2[example_id]['indices']
        score1 = scores1[example_id]['scores']
        score2 = scores2[example_id]['scores']

        concat_score = np.concatenate((score1, score2))
        concat_indices = np.concatenate((indices1, indices2))
        merged_indices = []
        for i in np.argsort(-concat_score):
            merged_indices.append(concat_indices[i])
        res[example_id]['scores'] = -np.sort(-concat_score)
        res[example_id]['indices'] = np.array(merged_indices)
    return res


def main(args):
    annotation_url = args.annotation_url.replace(
        '/edit#gid=', '/export?format=csv&gid=')
    annotations_df = pd.read_csv(annotation_url, on_bad_lines='skip')
    print(len(annotations_df))
    print(annotations_df)
    annotations = read_annotations(annotations_df, args.using_prediction)
    import_module_and_submodules("src.predictors")
    import_module_and_submodules("src.dataset_reader")
    import_module_and_submodules("src.model")

    if args.model == "nq-nli":
        model_path = "/mnt/data0/jfchen/qa_via_entailment/model_data/nq-nli-model.tar.gz"
        model_name = "qa_nli"
    elif args.model == 'mnli':
        model_path = "/mnt/data0/jfchen/pre-trained-models/mnli_roberta-2020.06.09.tar.gz"
        model_name = "textual_entailment"
    elif args.model == 'doc-nli':
        model_path = "/mnt/data0/jfchen/fine_grained_fact_verification/experiment_logs/doc_nli/model.tar.gz"
        model_name = "qa_nli"
    else:
        raise ValueError('no model named {}'.format(args.model))

    predictor = Predictor.from_path(
        model_path,
        model_name,
        cuda_device=0)

    nli_scores = get_nli_scores_claim(
        annotations,
        predictor=predictor
    )
    nli_performance = get_nli_performance(annotations, nli_scores, merged=True)
    print("NLI Claim: F1: {:.3f}, P: {:.3f}, R: {:.3f}".format(
        nli_performance[0], nli_performance[1], nli_performance[2])
    )
    for num in range(10):
        bm25_performance = get_bm25_performance(annotations, num)
        print("bm25: F1-question: {:.3f}, F1-claim: {:.3f}".format(
            bm25_performance[0], bm25_performance[1])
        )

    question_converter = PolarQuestionConverter()
    nli_scores_positive = get_nli_scores_question(
        annotations,
        predictor = predictor,
        question_converter=question_converter,
        do_negation=False
    )
    nli_scores_negative = get_nli_scores_question(
        annotations,
        predictor = predictor,
        question_converter=question_converter,
        do_negation=True
    )
    merged_nli_scores = merge_nli_scores(nli_scores_positive, nli_scores_negative)
    for score, merged in zip(
            [nli_scores_positive, nli_scores_negative, merged_nli_scores],
            [True, True, True]
    ):
        for num in range(1, 10):
            nli_performance = get_nli_performance(annotations, score, merged)
            print("Evidence num {} NLI Claim: F1: {:.3f}, P: {:.3f}, R: {:.3f}".format(
                num, nli_performance[0], nli_performance[1], nli_performance[2])
            )
    nli_performance = get_nli_performance_merged(annotations,
                                                 nli_scores_positive,
                                                 nli_scores_negative)
    print("NLI Claim: F1: {:.3f}, P: {:.3f}, R: {:.3f}".format(
        nli_performance[0], nli_performance[1], nli_performance[2])
    )
    human_performance = get_human_performance(annotations)
    print("human: F1: {:.3f}, P: {:.3f}, R: {:.3f}".format(
        human_performance[0], human_performance[1], human_performance[2])
    )
    random_performance = get_random_baseline_performance(annotations)
    print("random: F1: {:.3f}, P: {:.3f}, R: {:.3f}".format(
        random_performance[0], random_performance[1], random_performance[2])
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_url', type=str, default=None)
    parser.add_argument('--using_prediction', type=int, default=0)
    parser.add_argument('--model', type=str, default='mnli')
    parsed_args = parser.parse_args()
    main(parsed_args)
