"""Reproduce bounded lexical statistics from two official N3 practice workbooks.

Sources are fetched into memory; no question text or PDF is written to disk.
Requires PyMuPDF 1.26.5 and Janome 0.5.0. See N3_most_used_vocabulary.md.
"""
from pathlib import Path
import concurrent.futures, collections, hashlib, json, re, sys, unicodedata, urllib.request

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '.research_tools'))
import fitz
from janome.tokenizer import Tokenizer

# Function words, speaker labels, and recurring test/instruction terms.
STOP = set('する ある いる なる できる れる られる せる させる こと もの ところ ため よう これ それ あれ ここ そこ どこ こちら そちら どちら 何 なん 私 あなた 彼 彼女 自分 方 人 今 つぎ 次 問題 例 答え 回答 質問 文 文章 ことば 言葉 一つ 最も 番 読み方 男性 女性 男 女 一 二 三 四 五 六 七 八 九 十 そう どう この その あの どの ない 解 読 貸し'.split())
ALIASES = {'いい':'良い', 'よい':'良い', 'えらぶ':'選ぶ'}
META_LINE = re.compile(r'^(?:Language Knowledge|Notes|問題\s*\d+|例)$|最もよいもの|えらびなさい|選びなさい|^\(M:男性 F:女性\)$')

def get_document(item):
    year, part = item
    url = f'https://www.jlpt.jp/samples/sample{year}/pdf/N3{part}.pdf'
    data = urllib.request.urlopen(url, timeout=90).read()
    doc = fitz.open(stream=data, filetype='pdf')
    pages = []
    for page_no, page in enumerate(doc, 1):
        spans = []
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines', []):
                for span in line['spans']:
                    spans.append(span)
        # Vocabulary and grammar PDFs start with a cover and blank page.
        if year == 2018 and part in ('V', 'G') and page_no <= 2:
            pages.append({'pdf_page': page_no, 'included': False, 'text': '', 'characters': 0})
            continue
        # Body font sizes are checked per page. Small furigana is excluded.
        body_spans = [s for s in spans if (8.5 <= s['size'] <= 11.2 if part == 'script' else 10.0 <= s['size'] <= 16.0)]
        if part != 'script' and not any(abs(s['size'] - 11.3) < 0.2 for s in spans):
            pages.append({'pdf_page': page_no, 'included': False, 'text': '', 'characters': 0})
            continue
        # Sort by baseline then x. Furigana is excluded by font size.
        lines = collections.defaultdict(list)
        for s in body_spans:
            # Page footers and running headers are excluded geometrically too.
            # Exclude page furniture, including the vertical 読/解 label.
            if s['bbox'][0] < 50 or s['bbox'][1] < 25 or s['bbox'][3] > page.rect.height - 25:
                continue
            baseline = round(s['origin'][1] / 2) * 2
            lines[baseline].append(s)
        line_text = []
        for baseline in sorted(lines):
            line_text.append(''.join(s['text'] for s in sorted(lines[baseline], key=lambda x: x['bbox'][0])))
        text = unicodedata.normalize('NFKC', '\n'.join(x for x in line_text if not META_LINE.search(x.strip())))
        pages.append({'pdf_page': page_no, 'included': True, 'text': text, 'characters': len(text)})
    return {'id': f'{year}-{part}', 'url': url, 'sha256': hashlib.sha256(data).hexdigest(), 'pdf_pages': len(doc), 'pages': pages}

def main():
    tokenizer = Tokenizer()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        # 2012 V/G/R are image-only in this reproducible extraction and are not
        # OCRed. Its listening script is text-readable. All four 2018 text
        # components are included.
        items = [(2012, 'script')] + [(2018, p) for p in ('V','G','R','script')]
        documents = list(pool.map(get_document, items))
    occurrences = collections.defaultdict(collections.Counter)
    locations = collections.defaultdict(lambda: collections.defaultdict(set))
    readings = collections.defaultdict(collections.Counter)
    summaries = []
    for doc in documents:
        kept = 0
        for page in doc['pages']:
            for token in tokenizer.tokenize(page['text']):
                pos = token.part_of_speech.split(',')
                if pos[0] not in ('名詞','動詞','形容詞','副詞') or pos[1] in ('数','代名詞','非自立','接尾'):
                    continue
                lemma = token.base_form if token.base_form != '*' else token.surface
                lemma = ALIASES.get(lemma, lemma)
                if lemma in STOP or not re.search(r'[\u3040-\u30ff\u4e00-\u9fff]', lemma):
                    continue
                if len(lemma) == 1 and not re.search(r'[\u4e00-\u9fff]', lemma):
                    continue
                occurrences[lemma][doc['id']] += 1
                locations[lemma][doc['id']].add(page['pdf_page'])
                readings[lemma][token.reading] += 1
                kept += 1
        summaries.append({k:doc[k] for k in ('id','url','sha256','pdf_pages')} | {
            'retained_tokens': kept,
            'page_audit': [{k:p[k] for k in ('pdf_page','included','characters')} for p in doc['pages']]
        })
    ranked = sorted(occurrences, key=lambda w:(-sum(occurrences[w].values()), w))
    rows = []
    for word in ranked[:100]:
        rows.append({'word':word, 'reading_auto':readings[word].most_common(1)[0][0],
            'total':sum(occurrences[word].values()), 'counts':dict(occurrences[word]),
            'pdf_pages':{k:sorted(v) for k,v in locations[word].items()}})
    result = {'analysis_date':'2026-10-27', 'method_version':1, 'source_type':'2012 and 2018 compiled official practice workbooks, not dated exam sessions',
        'stopwords':sorted(STOP), 'documents':summaries, 'candidate_unique_lemmas':len(occurrences),
        'retained_tokens':sum(sum(c.values()) for c in occurrences.values()), 'top100_candidates':rows}
    (ROOT/'official_sample_counts.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'documents':len(documents),'retained_tokens':result['retained_tokens'],'candidate_unique_lemmas':len(occurrences),
        'top100':[(r['word'],r['total']) for r in rows], 'pages':[(s['id'],s['pdf_pages'],sum(p['included'] for p in s['page_audit'])) for s in summaries]},ensure_ascii=False))

if __name__ == '__main__':
    main()
