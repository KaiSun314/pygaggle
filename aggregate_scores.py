import string
import re
import argparse
from tqdm import tqdm
import numpy as np
import json
import logging
import coloredlogs
import os
from collections import defaultdict

coloredlogs.install(level='INFO',
                    fmt='%(asctime)s [%(levelname)s] %(module)s: %(message)s')

def parse(arg):
    settings = arg.split('/')
    retriever_method = settings[0]
    reader_method = settings[1]
    if len(settings) >= 4:
        betas = list(map(float, settings[2].split(',')))

        mult = True if settings[3][0] == '*' else False
        gammas = list(map(float, settings[3][1:].split(',')))
    else:
        betas = [None]
        gammas = [None]
        mult = None

    if len(settings) == 3:
        topk = list(map(int, settings[2].split(',')))
    elif len(settings) == 5:
        topk = list(map(int, settings[4].split(',')))
    else:
        topk = None

    return retriever_method, reader_method, betas, gammas, mult, topk

def exact_match_score(prediction, ground_truth):
    return _normalize_answer(prediction) == _normalize_answer(ground_truth)

def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

class RetrieverBase:
    def rank_topk(self, texts):
        pass

    def get_retriever_score(self):
        pass

class RetrieverHybrid(RetrieverBase):
    def rank_topk(self, texts):
        return texts

    def get_retriever_score(self):
        return lambda x: x['hybrid_score']

class RetrieverDPR(RetrieverBase):
    def rank_topk(self, texts):
        topk_texts = sorted(texts, reverse=True, key=lambda x: float(x['dpr_score']))
        for i, x in enumerate(topk_texts):
            if x['dpr_score'] == -1:
                return topk_texts[:i]

    def get_retriever_score(self):
        return lambda x: x['dpr_score']

class RetrieverBM25(RetrieverBase):
    def rank_topk(self, texts):
        topk_texts = sorted(texts, reverse=True, key=lambda x: float(x['bm25_score']))
        for i, x in enumerate(topk_texts):
            if x['bm25_score'] == -1:
                return topk_texts[:i]

    def get_retriever_score(self):
        return lambda x: x['bm25_score']

class ReaderBase:
    def __init__(self, retriever, beta, gamma):
        self.get_retriever_score = retriever.get_retriever_score()
        self.beta = beta
        self.gamma = gamma

    def reset_cache(self):
        pass

    def get_answer(self, texts):
        pass

class ReaderDPR(ReaderBase):
    def reset_cache(self):
        self.cache = {}

    def top_answer(self, texts, D):
        top_passage = max(texts + list(self.cache.values()), key=D)
        self.cache[0] = top_passage
        return top_passage['answers'][0]['answer']

    def get_answer(self, texts):
        return self.top_answer(
            texts,
            lambda x: float(x['reader_relevance_score'])
        )

class ReaderDPRFusion(ReaderDPR):
    def get_answer(self, texts):
        return super().top_answer(
            texts,
            lambda x: float(x['reader_relevance_score']) * self.beta + float(self.get_retriever_score(x)) * self.gamma
        )

class ReaderGAR(ReaderBase):
    def reset_cache(self):
        self.cache = defaultdict(int)

    def top_span_score(self, texts, D):
        eD = np.exp(np.array([D(text) for text in texts]))

        for i, text in enumerate(texts):
            text_answers = text['answers'][:5]
            eSi = np.exp(np.array([float(answer['span_score']) for answer in text_answers]))
            softmaxSi = list(eSi / np.sum(eSi))

            for j, answer in enumerate(text_answers):
                self.cache[_normalize_answer(answer['answer'])] += eD[i] * softmaxSi[j]

        return max(list(self.cache.items()), key=lambda x: x[1])[0]

    def get_answer(self, texts):
        return self.top_span_score(
            texts,
            lambda x: float(x['reader_relevance_score'])
        )

class ReaderGARFusion(ReaderGAR):
    def get_answer(self, texts):
        return super().top_span_score(
            texts,
            lambda x: float(x['reader_relevance_score']) * self.beta + float(self.get_retriever_score(x)) * self.gamma
        )

def aggregate_scores(predictions, retriever_method, reader_method, beta, gamma, topk, output_dir):
    retriever = globals()[f'Retriever{retriever_method}']()
    reader = globals()[f'Reader{reader_method}'](retriever, beta, gamma)

    ems = { k : [] for k in topk }
    for prediction in tqdm(predictions):
        reader.reset_cache()

        ground_truth_answers = prediction['ground_truth_answers']

        texts = retriever.rank_topk(prediction['contexts']);

        prevk = 0
        for k in topk:
            answer = reader.get_answer(texts[prevk: k])

            em_hit = max([exact_match_score(answer, ga) for ga in ground_truth_answers])
            ems[k].append(em_hit)

            prevk = k

    logging.info(f'retriever={retriever_method}, reader={reader_method}, beta={beta}, gamma={gamma}')
    for k in topk:
        em = np.mean(np.array(ems[k])) * 100.
        logging.info(f'Top{k}\tExact Match Accuracy: {em}')

    if output_dir is not None:
        for k in topk:
            with open(os.path.join(
                output_dir,
                f'{retriever_method}_{reader_method}_{k}' + (f'_{beta}_{gamma}' if beta is not None else '')
            ), 'w') as f:
                for em_hit in ems[k]:
                    f.write(('1' if em_hit else '0') + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions-file', type=str)
    parser.add_argument('--reader', type=str, nargs='+')
    parser.add_argument('--topk', type=int, nargs='+')
    parser.add_argument('--output-dir', type=str)
    args = parser.parse_args()

    logging.info('Loading the Predictions File')
    predictions = json.load(open(args.predictions_file))

    for arg in args.reader:
        retriever_method, reader_method, betas, gammas, mult, topk = parse(arg)

        if topk is None:
            topk = args.topk

        logging.info(f'Aggregating Using Retriever = {retriever_method} and Reader = {reader_method}')
        for beta in betas:
            for gamma in gammas:
                aggregate_scores(
                    predictions,
                    retriever_method,
                    reader_method,
                    beta,
                    (beta * gamma if mult else gamma),
                    topk,
                    args.output_dir
                )
