# Generating Literal and Implied Subquestions to Fact-check Complex Claims

This repository contains the data and code for the baseline described in the following paper:

> [**Generating Literal and Implied Subquestions to Fact-check Complex Claims**](https://arxiv.org/pdf/2205.06938.pdf)<br/>
> Jifan chen, Aniruddh Sriram, Eunsol Choi, Greg Durrett<br/>
> arxiv preprint
```
@article{chen-etal-2022-generating,
  title={Generating Literal and Implied Subquestions to Fact-check Complex Claims},
  author={Chen, Jifan and Sriram, Aniruddh and Choi, Eunsol and Durrett, Greg},
  journal={arxiv preprint},
  year={2022}
}
```

## Get Started
`git clone https://github.com/jifan-chen/subquestions-for-fact-checking.git`

Install the dependencies by running 
`pip install -r requirements.txt`

 
## Datasets

To download the dataset, simply run `bash scripts/download_data.sh`
The data files are located under `./ClaimDecomp`.

- `train.jsonl` contains 800 unique claims paired with the decomposed questions.
- `dev.jsonl` contains 200 unique claims paired with the decomposed questions.
- `test.jsonl` contains 200 unique claims paired with the decomposed questions.

The data files are formatted as jsonlines. Here is a single example:
```
{
    'example_id': '-7643898299150913613',
    'claim': 'With voting by mail, you get thousands and thousands of people sitting in somebody's living room, signing ballots all over the place.',
    'label': 'false',
    'person': 'Donald Trump',
    'venue': 'stated on April 7, 2020 in a press briefing:',
    'url': 'https://www.politifact.com/factchecks/2020/apr/09/donald-trump/donald-trumps-dubious-claim-thousands-are-conspiri/',
    'justification': 'Trump said that with voting by mail, "you get thousands and thousands of people sitting in somebody's living room, signing ballots all over the place. "Voting fraud in general is considered to be rare, although voting experts agree that the risks are greater for mail balloting than for in-person voting. Still, Trump didn't produce any evidence for the "thousands and thousands" claim, and voting experts said his assertion doesn't square with what is known about the actual cases of voting fraud in the recent past.\nWe rate the statement False.'
    'annotations':[
        {"questions": ["Is voting fraud widespread in the US?", "Is there a greater risk of voting fraud with mail-in ballots?", "Is there evidence of thousands of people committing mail-in voting fraud?"],
         "answers": ["no", "yes", "no"],
         "statements": ["Voting fraud is widespread in the US.", "There is a greater risk of voting fraud with mail-in ballots.", "There is evidence of thousands of people committing mail-in voting fraud."]
         "statements-negate": ["Voting fraud is not widespread in the US.", "There is no greater risk of voting fraud with mail-in ballots.", "There is no evidence of thousands of people committing mail-in voting fraud."]
        }
    ...
    ]
}
```

| Field            |    type     | Description                                                                              |
|------------------|-------------|------------------------------------------------------------------------------------------|
| `example_id`     |    string  | Example ID                                                                          |
| `claim`          |    string  | Claim                                                                                    |
| `label`          |    string  | Label: pants-fire, false, barely-true, half-true, mostly-true, true                      |
| `person`         |    string  | Person who made the claim                                                                |
| `venue`          |    string  | Date and venue of the claim                                                              |
| `url`            |    string  | Politifact url of the claim                                                              |
| `justification`  |    string  | Justification paragraph writen by the fact-checkers                                      |
| `full_article`   |    string  | Full verification article writen by the fact-checkers                                    |
| `annotations`    |    List[dict]    | Annotation of our decomposed questions                                             |

Each `annotation` is formatted as follows:

| Field            |    type     | Description                                                                              |
|------------------|-------------|------------------------------------------------------------------------------------------|
| `questions`         |  List[string]  | Yes-no questions related to checking the veracity of the claim                   |
| `answers`           |  List[string]  | Answer to the question: yes/no/unknown                   |
| `question_source`   |  List[string]  | Question source: claim or justification
| `statements`        |  List[string]  | Statements converted from the yes-no questions                                   |
| `statements_negate` |  List[string]  | Negated statements    |

## Running NLI models
To run the three NLI models -- `NQ-NLI`, `Doc-NLI`, `MNLI` trained in our paper, simply run `bash scripts/run_nli_models.sh`. You will need to install `allennlp==2.7.0` and `torch==1.9.0`. Check `scripts/run_nli_models.sh` for details about where the models are downloaded and how to switch models. 

## Question Generator
Coming soon ...

## Contact 

Please contact at `jfchen@cs.utexas.edu` if you have any questions.