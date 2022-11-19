import requests
import pandas as pd
import re
import unicodedata
import argparse
from bs4 import BeautifulSoup, Tag, NavigableString
from tqdm import tqdm

RULINGS = ['true', 'mostly-true', 'half-true', 'barely-true', 'false',
           'pants-fire']
RULINGS_IN_TEXT = ['true', 'mostly true', 'half true', 'mostly false', 'false',
                   'pants on fire', 'barely true']
RULING_SEC_PATTERN = "Our [R,r][uling,ating]"


def extract_all_paragraphs(paras):
    evidence = []
    for para in paras:
        if para is not None \
                and not isinstance(NavigableString, NavigableString) \
                and not para == '\n':
            child_num = len(para.find_all())
            # no child other than hyperlinks
            if para.name == 'table' or (para.name == 'div' and child_num >= 2):
                pass
            else:
                text = para.get_text(strip=True)
                if text:
                    evidence.append(unicodedata.normalize(
                        "NFKC",
                        text)
                    )
    return evidence


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
        # find content
        claim = soup.find('div', attrs={"class": "m-statement__quote"})
        # find the full article
        full_article = soup.find('article', attrs={"class": 'm-textblock'})
        # find when & where
        time_venue = soup.find('div', attrs={"class": "m-statement__desc"})
        time_venue = time_venue.get_text(strip=True)
        name = name.get_text(strip=True)
        claim = claim.get_text(strip=True)
        paras = full_article.find_all('p')
        full_article, ruling = extract_all_paragraphs(paras)
        anchor = soup.find('div', text=re.compile(RULING_SEC_PATTERN),
                           recursive=True)

        if not anchor:
            anchor = soup.find('strong', text=re.compile(RULING_SEC_PATTERN),
                               recursive=True)
        if not anchor:
            anchor = soup.find('p', text=re.compile(RULING_SEC_PATTERN),
                               recursive=True)
        while get_sibling(anchor) is None or get_sibling(anchor) == '\n':
            anchor = anchor.parent
        justification_para = []
        anchor = get_sibling(anchor)
        while isinstance(anchor, Tag):
            if not anchor.find('p') and not anchor.name == 'p':
                anchor = get_sibling(anchor)
                continue
            paras = anchor.find_all('p')
            if not paras:
                paras = [anchor]
            for para in paras:
                if para is not None \
                        and not isinstance(NavigableString, NavigableString) \
                        and not para == '\n':
                    justification_para.append(unicodedata.normalize(
                        "NFKC",
                        para.get_text(strip=True)
                    )
                    )
            anchor = get_sibling(anchor)

        while justification_para[-1] != full_article[-1]:
            justification_para.pop()
        for i in range(len(justification_para)):
            full_article.pop()

        assert claim
        assert name
        assert time_venue
        assert full_article
        assert justification_para
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
