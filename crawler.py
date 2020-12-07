import urllib.request as ul
from bs4 import BeautifulSoup
from tqdm import tqdm
import pickle
import os
from multiprocessing import Pool, cpu_count
import numpy as np

def parse(line):
    l = []
    header = {'User-Agent' : 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'}
    line = line.split('\t')
    user_id = line[0]
    url = 'https://en.wikipedia.org/wiki/' + line[1][8:].strip()
    try:
        count = {}
        req = ul.Request(url, headers=header)
        page = ul.urlopen(req).read()
        soup = BeautifulSoup(page, 'lxml')
        for category in soup.find('div', {'id': 'mw-normal-catlinks'}).find('ul').find_all('li'):
            category = category.find('a').text
            l.append(category)
        for cat in l:
            for word in cat.split():
                if word not in stopwords:
                    if word in count:
                        count[word] += 1
                    else:
                        count[word] = 0
    except:
        pass
    return user_id, sorted(count, key=count.get, reverse=True)

def one_hot_encoding(set, size):
    one_hot_vector = np.zeros(size, np.int32)
    for value in set:
        one_hot_vector[value] = 1
    return one_hot_vector

if __name__ == '__main__':
    f = open('friend-based_interest_info.tsv', 'r', encoding='ISO-8859-1')
    lines = f.readlines()

    s = open('english.txt', 'r')
    stopwords = set()
    for line in s.readlines():
        stopwords.add(line.strip())

    TMP_DIR = 'tmp/'
    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)

    preferencies = {}
    categories = {}
    index = 0
    most_common = 3

    p = Pool(processes=cpu_count())
    for result in tqdm(p.imap(parse, lines), total=len(lines)):

        user_id, category = result
        category = category[:most_common]

        for cat in category:
            if cat not in categories:
                categories[cat] = index
                index += 1

        if user_id not in preferencies:
            preferencies[user_id] = set()
        preferencies[user_id].update([categories[cat] for cat in category])
    p.close()
    p.join()

    categories_lenght = len(categories)
    del categories

    print('Categories lenght:', categories_lenght)

    for user_id in preferencies:
        preferencies[user_id] = one_hot_encoding(preferencies[user_id], categories_lenght)

    pickle.dump(preferencies, open(TMP_DIR + 'preferencies.pkl', 'wb'))