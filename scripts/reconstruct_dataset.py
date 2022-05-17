import requests
import pandas as pd
import re
import unicodedata
import argparse
from bs4 import BeautifulSoup, Tag, NavigableString
from tqdm import tqdm

RULINGS = ['true', 'mostly-true', 'half-true', 'barely-true', 'false', 'pants-fire']
RULINGS_IN_TEXT = ['true', 'mostly true', 'half true', 'mostly false', 'false', 'pants on fire', 'barely true']
RULING_SEC_PATTERN = "Our [R,r][uling,ating]"


def extract_all_paragraphs(start_node):
    node = start_node
    finder = re.compile(RULING_SEC_PATTERN)
    evidence = []
    while node is not None and not finder.match(node.get_text(strip=True)):
        child_num = len(node.find_all())
        # no child other than hyperlinks
        if node.name == 'table' or (node.name == 'div' and child_num >= 2):
            pass
        else:
            text = node.text.strip()
            if text:
                evidence.append(text)
        node = get_sibling(node)
    return '\n'.join(evidence)


def get_sibling(element):
    sibling = element.next_sibling
    if sibling == "\n" or isinstance(sibling, NavigableString):
        return get_sibling(sibling)
    else:
        return sibling


def main(args):
    df = pd.read_json(args.input_path, lines=True)
    output = []
    for i, row in tqdm(df.iterrows()):
        page_url = row['url']
        r = requests.get(page_url)
        soup = BeautifulSoup(r.text, 'html.parser')
        name = soup.find(attrs={"class": "m-statement__name"})
        # find when & where
        time_venue = soup.find('div', attrs={"class": "m-statement__desc"})
        # find content
        claim = soup.find('div', attrs={"class": "m-statement__quote"})
        # find the full article
        full_article = soup.find('article', attrs={"class": 'm-textblock'})
        if not (name and time_venue and full_article and claim):
            return
        time_venue = soup.find('div', attrs={"class": "m-statement__desc"})
        time_venue = time_venue.get_text(strip=True)
        name = name.get_text(strip=True)
        claim = claim.get_text(strip=True)
        first_para = full_article.find('p')
        # print(name)
        # print(time_venue)
        # print(claim)
        # print(full_article)
        full_article = extract_all_paragraphs(first_para)
        anchor = soup.find('div', text=re.compile(RULING_SEC_PATTERN))
        if not anchor:
            anchor = soup.find('strong', text=re.compile(RULING_SEC_PATTERN))
        if not anchor:
            return
        if not anchor.next_sibling:
            anchor = anchor.parent
        # print(anchor)
        assert name and claim and time_venue and full_article
        justification_para = []
        anchor = get_sibling(anchor)
        while isinstance(anchor, Tag):
            anchor_text = anchor.text.strip()
            if anchor.name == 'p' and anchor_text:
                justification_para.append(unicodedata.normalize(
                    "NFKC",
                    anchor.get_text(strip=True)
                    )
                )
            anchor = get_sibling(anchor)
        assert justification_para
        justification_para = "\n".join(justification_para)
        row['claim'] = claim
        row['person'] = name
        row['venue'] = time_venue
        row['justification'] = justification_para
        row['full_article'] = full_article
        output.append(row)

    output_frame = pd.DataFrame(output)
    output_frame.to_json(args.output_path, orient='records', lines=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str,
                        help='output path to processed dataset')
    main(parser.parse_args())
