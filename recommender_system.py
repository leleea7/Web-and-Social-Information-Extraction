import multiprocessing as mp
import numpy as np
import urllib.request as ul
from bs4 import BeautifulSoup

class RecommenderSystem():

    def __init__(self, preferencies, categories, s23_file):
        self.preferencies = preferencies
        self.categories = categories
        self.proposed_pages = {}
        for line in s23_file:
            line = line.strip().split('\t')
            try:
                self.proposed_pages[line[0]].append(line[1][8:])
            except:
                self.proposed_pages[line[0]] = [line[1][8:]]
        
    def __parse(self, item):
        l = []
        header = {'User-Agent' : 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'}
        url = 'https://en.wikipedia.org/wiki/' + item
        try:
            req = ul.Request(url, headers=header)
            page = ul.urlopen(req).read()
            soup = BeautifulSoup(page, 'html.parser')
            for category in soup.find('div', {'id': 'mw-normal-catlinks'}).find('ul').find_all('li'):
                category = category.find('a').text
                l.append(category)
        except:
            pass
        return l
    
    def cosine_similarity(self, v, w):
        assert(len(v) == len(w))
        numerator = np.sum(v * w)
        denominator = np.sqrt(np.sum(np.square(v))) * np.sqrt(np.sum(np.square(w)))
        return numerator / denominator

    def euclidean_distance(self, v, w):
        assert(len(v) == len(w))
        return np.sqrt(np.sum(np.square(v - w)))

    def pearson_correlation(self, v, w):
        assert(len(v) == len(w))
        mean_v = np.mean(v)
        mean_w = np.mean(w)
        numerator = np.sum((v - mean_v) * (w - mean_w))
        denominator = np.sqrt(np.sum(np.square(v - mean_v)) * np.sum(np.square(w - mean_w)))
        return numerator / denominator
    
    def most_relevant_pages(self, user_id, metrics='all'):
        if metrics == 'all':
            scores_1 = []
            scores_2 = []
            scores_3 = []
        else:
            scores = []
        v = np.zeros(len(self.categories))
        for page in self.proposed_pages[user_id]:
            cat = self.__parse(page)
            for c in cat:
                if c in self.categories:
                    v[self.categories[c]] = 1
            if metrics == 'all':
                score_1 = self.cosine_similarity(self.preferencies[user_id], v)
                score_2 = self.euclidean_distance(self.preferencies[user_id], v)
                score_3 = self.pearson_correlation(self.preferencies[user_id], v)
                scores_1.append((score_1, page))
                scores_2.append((score_2, page))
                scores_3.append((score_3, page))
            elif metrics == 'cosine':
                score = self.cosine_similarity(self.preferencies[user_id], v)
                scores.append((score, page))
            elif metrics == 'euclidean':
                score = self.euclidean_distance(self.preferencies[user_id], v)
                scores.append((score, page))
            elif metrics == 'pearson':
                score = self.pearson_correlation(self.preferencies[user_id], v)
                scores.append((score, page))
        if metrics == 'all':
            scores_1.sort(key=lambda x: x[0], reverse=True)
            scores_2.sort(key=lambda x: x[0])
            scores_3.sort(key=lambda x: x[0], reverse=True)
            print('Top 3 (cosine similarity): ', [x[1] for x in scores_1[:3]])
            print('Top 3 (euclidean distance): ', [x[1] for x in scores_2[:3]])
            print('Top 3 (pearson correlation): ', [x[1] for x in scores_3[:3]])
            return None
        if metrics == 'euclidean':
            scores.sort(key=lambda x: x[0])
        else:
            scores.sort(key=lambda x: x[0], reverse=True)
        return scores