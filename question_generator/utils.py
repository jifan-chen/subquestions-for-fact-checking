from typing import Text, List, Dict
import json
import csv

SEP_TK = "###"
EMPTY_ANNOTATION = "{}"
INVALID_PREDICTION = "INVALID"


def form_decontext_train_input(paragraph_text: Text,
                               answer_sentence: Text,
                               page_tile: Text = None
                               ):
    marked_answer_sent = "</s> {} </s>".format(answer_sentence)
    paragraph_text = paragraph_text.replace(
        answer_sentence,
        marked_answer_sent
    )
    paragraph_text = "{} </s> {}".format(page_tile, paragraph_text)
    return paragraph_text


def form_decontext_train_output(answer_sentence: Text,
                                decontext_sentence: Text,
                                category: Text
                                ):
    if category == 'DONE':
        return "DONE ### {}".format(decontext_sentence)
    else:
        return "{} ### {}".format(category, answer_sentence)


def form_esnil_train_output(label: Text,
                            spans_text: List[Text],
                            explanation: Text):
    output = label
    for sp_text in spans_text:
        output = "{} {} {}".format(output, SEP_TK, sp_text)
    output = "{} {} {}".format(output, SEP_TK, explanation)
    return output


def write_decontext_predictions_out(dataset,
                                    predictions: List[str],
                                    output_path: str,
                                    output_format: str = None,
                                    data_source: str = None
                                    ):
    if output_format == "csv":
        csv_file = open(output_path, 'w', newline='')
        csv_writer = csv.writer(csv_file, delimiter=',')
        if data_source == 'qa-nli':
            csv_fields = ['example_id', 'question', 'question_statement',
                          'answer_sent', 'decontext_answer_sent', 'paragraph']
            csv_writer.writerow(csv_fields)
            for data, pred in zip(dataset, predictions):
                csv_writer.writerow([data['example_id'],
                                     data['question_text'],
                                     data['question_statement_text'],
                                     data['answer_sent_text'],
                                     pred,
                                     data['paragraph_text']])
    else:
        with open(output_path, 'w') as fout:
            for data, pred in zip(dataset, predictions):
                try:
                    cat, sent = pred.split('###')
                except ValueError:
                    cat = 'IMPOSSIBLE'
                    sent = data['answer_sent_text']
                if data_source == 'qa-nli':
                    data['decontext_answer_sent_text'] = sent.strip()
                else:
                    data['decontextualized_sentence'] = sent.strip()
                data['category'] = cat.strip()
                json.dump(data, fout)
                fout.write('\n')


def write_question_converter_predictions_out(dataset,
                                             predictions: List[str],
                                             output_path: str,
                                             output_format: str = None,
                                             data_source: str = None
                                             ):
    if output_format == 'csv':
        csv_file = open(output_path, 'w', newline='')
        csv_writer = csv.writer(csv_file, delimiter=',')
        if data_source == 'qa-nli':
            csv_fields = ['example_id', 'question', 'answer', 'question_statement']
            csv_writer.writerow(csv_fields)
            for data, pred in zip(dataset, predictions):
                csv_writer.writerow([data['example_id'],
                                     data['question_text'],
                                     data['answer_text'],
                                     pred])
        else:
            csv_fields = ['question', 'answer', 'question_statement', 'turker_answer']
            csv_writer.writerow(csv_fields)
            for data, pred in zip(dataset, predictions):
                csv_writer.writerow([data['question'],
                                     data['answer'],
                                     pred,
                                     data['turker_answer']])
    else:
        with open(output_path, 'w') as fout:
            for data, pred in zip(dataset, predictions):
                data['question_statement_text'] = pred
                json.dump(data, fout)
                fout.write('\n')


def write_esnli_predictions_out(dataset,
                                predictions: List[str],
                                output_path: str,
                                output_format: str = None,
                                data_source: str = None
                                ):
    if output_format == 'csv':
        csv_file = open(output_path, 'w', newline='')
        csv_writer = csv.writer(csv_file, delimiter=',')
        if data_source == 'qa-nli':
            csv_fields = [
                'question_statement', 'decontext_answer_sent',
                'highlights1', 'highlights2', 'explanation'
            ]
            csv_writer.writerow(csv_fields)
            for data, pred in zip(dataset, predictions):
                items = pred.split('###')
                if len(items) < 4:
                    high_lights1 = INVALID_PREDICTION
                    high_lights2 = INVALID_PREDICTION
                    explanation = INVALID_PREDICTION
                else:
                    label, high_lights1, \
                        high_lights2, explanation = pred.split('###')

                csv_writer.writerow([data['converted_question'],
                                     data['decontext_answer_sent'],
                                     high_lights1,
                                     high_lights2,
                                     explanation])
        else:
            csv_fields = ['sentence1', 'sentence2', 'highlights1',
                          'highlights2', 'explanation']
            csv_writer.writerow(csv_fields)
            for data, pred in zip(dataset, predictions):
                label, high_lights1, high_lights2, explanation = pred.split('###')
                csv_writer.writerow([data['Sentence1'],
                                     data['Sentence2'],
                                     high_lights1,
                                     high_lights2,
                                     explanation])
    else:
        with open(output_path, 'w') as fout:
            for data, pred in zip(dataset, predictions):
                label, high_lights1, \
                    high_lights2, explanation = pred.split('###')
                data['highlights1'] = high_lights1
                data['highlights2'] = high_lights2
                data['pred_label'] = label
                data['explanation'] = explanation
                json.dump(data, fout)
                fout.write('\n')


def write_question_generator_predictions_out(
        dataset,
        predictions: List[str],
        output_path: str,
        output_format: str = None,
        data_source: str = None
        ):
    if output_format == 'csv':
        csv_file = open(output_path, 'w', newline='')
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_fields = ['sub-claim', 'question']
        csv_writer.writerow(csv_fields)
        for data, pred in zip(dataset, predictions):
            print('sub-claim: {}\n\nquestion: {}\n\n********'.format(data['sub-claim'], pred))
            csv_writer.writerow([data['sub-claim'], pred])
    else:
        with open(output_path, 'w') as fout:
            for data, pred in zip(dataset, predictions):
                print('sub-claim: {}\n\nquestion: {}\n\n********'.format(data['sub-claim'], pred))
                data['question'] = pred
                json.dump(data, fout)
                fout.write('\n')


def write_all_question_generator_ruling_predictions_out(
        dataset,
        predictions: List[str],
        output_path: str,
        output_format: str = None,
        data_source: str = None
        ):
    if output_format == 'csv':
        csv_file = open(output_path, 'w', newline='')
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_fields = ['claim', 'pred-questions', 'gold-questions']
        csv_writer.writerow(csv_fields)
        for data, pred in zip(dataset, predictions):
            print('claim: {}\n\nquestion: {}\n\nruling: {}\n\n********'.format(data['Input.claim'], pred, data['Input.evidence']))
            csv_writer.writerow(
                [data['Input.who'] + ' ' + data['Input.when_where'] + ' ' + data['Input.claim'] + SEP_TK +
                 data['Input.evidence'],
                 '\n'.join(pred.split(SEP_TK)),
                 data['golds']])
    else:
        with open(output_path, 'w') as fout:
            for data, pred in zip(dataset, predictions):
                data['question'] = pred
                json.dump(data, fout)
                fout.write('\n')


def write_all_question_generator_predictions_out(
        dataset,
        predictions: List[str],
        output_path: str,
        output_format: str = None,
        data_source: str = None
        ):
    if output_format == 'csv':
        csv_file = open(output_path, 'w', newline='')
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_fields = ['claim', 'pred-questions', 'gold-questions']
        csv_writer.writerow(csv_fields)
        for data, pred in zip(dataset, predictions):
            print('claim: {}\n\nquestion: {}\n\n********'.format(data['Input.claim'], pred))
            # data_list = json.loads(data['Answer.user-output'])
            csv_writer.writerow(
                [data['Input.who'] + ' ' + data['Input.when_where'] + ' ' + data['Input.claim'],
                 '\n'.join(pred.split(SEP_TK)),
                 data['golds']])
    else:
        with open(output_path, 'w') as fout:
            for data, pred in zip(dataset, predictions):
                data['question'] = pred
                json.dump(data, fout)
                fout.write('\n')


def write_one_question_generator_ruling_predictions_out(
        dataset,
        predictions: List[str],
        output_path: str,
        output_format: str = None,
        data_source: str = None
        ):
    if output_format == 'csv':
        csv_file = open(output_path, 'w', newline='')
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_fields = ['claim', 'pred-questions', 'gold-questions']
        csv_writer.writerow(csv_fields)
        for data, pred in zip(dataset, predictions):
            print('claim: {}\n\nquestion: {}\n\n********'.format(data['Input.claim'], pred))
            # data_list = json.loads(data['Answer.user-output'])
            golds = data['golds'].split('\n')
            golds = [q for q in golds if q]
            csv_writer.writerow(
                [str(len(golds)) + SEP_TK +
                 data['Input.who'] + ' ' + data['Input.when_where'] + ' ' + data['Input.claim'] + SEP_TK +
                 data['Input.evidence'],
                 '\n'.join(pred.split(SEP_TK)),
                 data['golds']])
    else:
        with open(output_path, 'w') as fout:
            for data, pred in zip(dataset, predictions):
                data['question'] = pred
                json.dump(data, fout)


def write_one_question_generator_predictions_out(
        dataset,
        predictions: List[str],
        output_path: str,
        output_format: str = None,
        data_source: str = None
        ):
    if output_format == 'csv':
        csv_file = open(output_path, 'w', newline='')
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_fields = ['claim', 'pred-questions', 'gold-questions']
        csv_writer.writerow(csv_fields)
        for data, pred in zip(dataset, predictions):
            print('claim: {}\n\nquestion: {}\n\n********'.format(data['Input.claim'], pred))
            # data_list = json.loads(data['Answer.user-output'])
            pred = set(pred.split(SEP_TK))
            golds = data['golds'].split('\n')
            golds = [q for q in golds if q]
            csv_writer.writerow(
                [str(len(golds)) + SEP_TK +
                 data['Input.who'] + ' ' + data['Input.when_where'] + ' ' + data['Input.claim'],
                 '\n'.join(pred),
                 data['golds']])
    else:
        with open(output_path, 'w') as fout:
            for data, pred in zip(dataset, predictions):
                data['question'] = pred
                json.dump(data, fout)


def process_decontext_train_and_dev(examples: Dict):
    inputs = []
    targets = []
    for para, answer_sent, title, annots in zip(
            examples['paragraph_text'],
            examples['original_sentence'],
            examples['page_title'],
            examples['annotations']
    ):
        if not annots:
            continue
        inputs.append(
            form_decontext_train_input(para, answer_sent, title)
        )
        # for training and a preliminary evaluation
        first_annot = annots[0]
        decontext_sent = first_annot['decontextualized_sentence']
        category = first_annot['category']
        targets.append(form_decontext_train_output(answer_sent,
                                                   decontext_sent,
                                                   category))
    return inputs, targets


def process_decontext_qanli(examples: Dict):
    inputs = []
    targets = []
    for para, answer_sent, title in zip(
            examples['paragraph_text'],
            examples['answer_sent_text'],
            examples['title_text']
    ):
        # if not para or not answer_sent or not title:
        #     print(para)
        #     print(answer_sent)
        #     print(title)
        #     input()
        inputs.append(
            form_decontext_train_input(para, answer_sent, title)
        )
        # for training and a preliminary evaluation
        targets.append("DUMB LABEL")
    return inputs, targets


def process_question_converter_train_and_dev(examples: Dict):
    inputs = []
    targets = []
    for a, q, t in zip(examples['answer'],
                       examples['question'],
                       examples['turker_answer']):
        if a and q and t:
            inputs.append(
                "{} </s> {}".format(q, a)
            )
            targets.append(t)
    return inputs, targets


def process_question_converter_qanli(examples: Dict):
    inputs = []
    targets = []
    for a, q in zip(examples['answer_text'],
                    examples['question_text']):
        if a and q:
            inputs.append(
                "{} </s> {}".format(q, a)
            )
            targets.append('DUMB LABEL')
    return inputs, targets


def process_esnli_train_and_dev(examples: Dict):
    inputs = []
    targets = []
    # TODO: try all annotations
    for s1, s2, label, sp1h1, sp2h1, explain1 \
            in zip(examples['Sentence1'],
                   examples['Sentence2'],
                   examples['gold_label'],
                   examples['Sentence1_Highlighted_1'],
                   examples['Sentence2_Highlighted_1'],
                   examples['Explanation_1']):
        if s1 and s2 and sp1h1 and sp2h1:
            inputs.append(
                "{} </s> {}".format(s1, s2)
            )
            # concat all the highlights for now
            # TODO: try separating the non-consecutive highlights
            s1 = s1.split()
            s2 = s2.split()

            sp1h1 = sorted(sp1h1.split(','))
            sp2h1 = sorted(sp2h1.split(','))

            if sp1h1[0] != EMPTY_ANNOTATION:
                sp1h1 = [int(i) for i in sp1h1 if int(i) < len(s1)]
                sp1h1_text = " ".join([s1[i] for i in sp1h1])
            else:
                sp1h1_text = "EMPTY"

            if sp2h1[0] != EMPTY_ANNOTATION:
                sp2h1 = [int(i) for i in sp2h1 if int(i) < len(s2)]
                sp2h1_text = " ".join([s2[i] for i in sp2h1])
            else:
                sp2h1_text = "EMPTY"

            targets.append(form_esnil_train_output(label,
                                                   [sp1h1_text, sp2h1_text],
                                                   explain1)
                           )

    return inputs, targets


def process_esnli_qanli(examples: Dict):
    inputs = []
    targets = []
    for hypothesis, premise in zip(
            examples['converted_question'],
            examples['decontext_answer_sent']
    ):
        inputs.append(
            "{} </s> {}".format(premise, hypothesis)
        )
        # for training and a preliminary evaluation
        targets.append("DUMB LABEL")
    return inputs, targets


def process_question_generator_train_and_dev(examples: Dict):
    inputs = []
    targets = []
    for sub_claim, question in zip(examples['sub-claim'],
                                   examples['consented-question']):
        if sub_claim and question:
            # inputs.append(sub_claim.replace("] [", "</s>").replace("[", "</s>").replace("]", "</s>"))
            inputs.append(sub_claim)
            targets.append(question)
    return inputs, targets


def process_all_question_generator_train_and_dev(examples: Dict,
                                                 worker_ids: set = None):
    inputs = []
    targets = []
    for exp_id, worker_id, who, when_where, claim, output in \
            zip(examples['Input.id'],
                examples['WorkerId'],
                examples['Input.who'],
                examples['Input.when_where'],
                examples['Input.claim'],
                examples['Answer.user-output']):
        if worker_ids and worker_id not in worker_ids:
            continue
        data = json.loads(output)
        if claim and data:
            inputs.append(who + ' ' + when_where + ' ' + claim)
            targets.append(SEP_TK.join([item['question'] for item in data]))
    return inputs, targets


def process_all_question_generator_ruling_train_and_dev(examples: Dict,
                                                        worker_ids: set = None):
    inputs = []
    targets = []
    for exp_id, worker_id, who, when_where, claim, output, ruling in \
            zip(examples['Input.id'],
                examples['WorkerId'],
                examples['Input.who'],
                examples['Input.when_where'],
                examples['Input.claim'],
                examples['Answer.user-output'],
                examples['Input.evidence']):
        if worker_ids and worker_id not in worker_ids:
            continue
        data = json.loads(output)
        if claim and data and ruling:
            inputs.append(who + ' ' + when_where + ' ' + claim + SEP_TK + ruling)
            targets.append(SEP_TK.join([item['question'] for item in data]))
    return inputs, targets


def process_qg_train_and_dev(
        examples: Dict,
        worker_ids: set = None,
        training: bool = True,
        sampling: bool = True):
    inputs = []
    targets = []
    if training:
        for exp_id, worker_id, who, when_where, claim, output in \
                zip(examples['Input.id'],
                    examples['WorkerId'],
                    examples['Input.who'],
                    examples['Input.when_where'],
                    examples['Input.claim'],
                    examples['Answer.user-output']):
            if worker_ids and worker_id not in worker_ids:
                continue
            data = json.loads(output)
            if claim and data:
                if sampling:
                    for item in data:
                        inputs.append(who + ' ' + when_where + ' ' + claim)
                        targets.append(item['question'])
                else:
                    inputs.append(str(len(data)) + SEP_TK + who + ' ' + when_where + ' ' + claim)
                    targets.append(SEP_TK.join([item['question'].strip() for item in data]))
    else:
        for who, when_where, claim, golds in \
            zip(examples['Input.who'],
                examples['Input.when_where'],
                examples['Input.claim'],
                examples['golds']):
            golds = golds.split('\n')
            golds = [q for q in golds if q]
            # inputs.append(who + ' ' + when_where + ' ' + claim)
            inputs.append(
                str(10) + SEP_TK + who + ' ' + when_where + ' ' + claim)
            targets.append(SEP_TK.join(golds))

    return inputs, targets


def process_qg_ruling_train_and_dev(
        examples: Dict,
        worker_ids: set = None,
        training: bool = True,
        sampling: bool = True):
    inputs = []
    targets = []
    if training:
        for exp_id, worker_id, who, when_where, claim, ruling, output in \
                zip(examples['Input.id'],
                    examples['WorkerId'],
                    examples['Input.who'],
                    examples['Input.when_where'],
                    examples['Input.claim'],
                    examples['Input.evidence'],
                    examples['Answer.user-output']):
            if worker_ids and worker_id not in worker_ids:
                continue
            data = json.loads(output)
            if claim and data and ruling:
                if sampling:
                    for item in data:
                        inputs.append(who + ' ' + when_where + ' ' + claim + SEP_TK + ruling)
                        targets.append(item['question'])
                else:
                    inputs.append(str(len(data)) + SEP_TK + who + ' ' + when_where + ' ' + claim + SEP_TK + ruling)
                    targets.append(SEP_TK.join([item['question'] for item in data]))
    else:
        for who, when_where, claim, ruling, golds in \
            zip(examples['Input.who'],
                examples['Input.when_where'],
                examples['Input.claim'],
                examples['Input.evidence'],
                examples['golds']):
            golds = golds.split('\n')
            golds = [q for q in golds if q]
            inputs.append(
                str(len(golds)) + SEP_TK + who + ' ' + when_where + ' ' + claim +
                SEP_TK + ruling)
            targets.append(SEP_TK.join(golds))

    return inputs, targets
