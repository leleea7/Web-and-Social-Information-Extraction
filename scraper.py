import urllib.request as ul
from bs4 import BeautifulSoup
from tqdm import tqdm
import pickle
import os
import multiprocessing as mp
import numpy as np

def parse(line):
    l = []
    header = {'User-Agent' : 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'}
    line = line.split('\t')
    user_id = line[0]
    url = 'https://en.wikipedia.org/wiki/' + line[1][8:].strip()
    try:
        req = ul.Request(url, headers=header)
        page = ul.urlopen(req).read()
        soup = BeautifulSoup(page, 'lxml')
        for category in soup.find('div', {'id': 'mw-normal-catlinks'}).find_all('li'):
            category = category.find('a').text
            l.append(category)
    except:
        #print(url)
        pass
    return user_id, l

def one_hot_encoding(set, size):
    one_hot_vector = np.zeros(size, np.int32)
    for value in set:
        one_hot_vector[value] = 1
    return one_hot_vector

if __name__ == '__main__':
    f = open('dataset/S22_preferences.tsv', 'r')
    lines = f.readlines()

    TMP_DIR = 'tmp/'
    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)

    preferencies = {}
    categories = {}
    index = 0

    p = mp.Pool(processes=mp.cpu_count())
    for result in tqdm(p.imap(parse, lines), total=len(lines), desc='Scraping data'):

        user_id, category = result

        for cat in category:
            if cat not in categories:
                categories[cat] = index
                index += 1

        if user_id not in preferencies:
            preferencies[user_id] = set()
        preferencies[user_id].update([categories[cat] for cat in category])

    categories_lenght = len(categories)

    print('Categories lenght:', categories_lenght)

    for user_id in preferencies:
        preferencies[user_id] = one_hot_encoding(preferencies[user_id], categories_lenght)

    pickle.dump(preferencies, open(TMP_DIR + 'preferencies.pkl', 'wb'))
    pickle.dump(categories, open(TMP_DIR + 'categories.pkl', 'wb'))
