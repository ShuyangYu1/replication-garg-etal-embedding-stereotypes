import csv
import numpy as np
from sklearn.decomposition import PCA
import sys
from io import StringIO

def load_yr(yr, loc):
    vectors = np.load('{}{}-w.npy'.format(loc, yr), allow_pickle=True)
    words = np.load('{}{}-vocab.pkl'.format(loc, yr), allow_pickle=True, encoding='latin1')
    counts = np.load('{}{}{}-counts.pkl'.format(loc, 'counts/', yr), allow_pickle=True, encoding='latin1')
    return vectors, words, counts

def save_files(yrs, oldloc, newloc, label):
    for yr in yrs:
        print('\n')
        print(yr)
        vectors, words, counts = load_yr(yr, oldloc)
        with open('{}vectors_{}{}.txt'.format(newloc, label, yr), 'w', newline='') as f:
            with open('{}/vocab/vocab_{}{}.txt'.format(newloc, label, yr), 'w', newline='') as f2:
                csvwriter = csv.writer(f, delimiter=' ')
                csvwritervoc = csv.writer(f2, delimiter=' ')
                for en in range(0, len(vectors)):
                    try:
                        word = words[en]
                        if isinstance(word, bytes):
                            word = word.decode('utf-8')
                        row = [word]
                        row.extend(vectors[en])
                        csvwriter.writerow(row)
                        csvwritervoc.writerow([word, counts[word]])
                    except Exception as e:
                        print(word, counts.get(word, None), e)

loc = '../vectors/coha/sgns/'
yrs = list(range(1910, 2000, 10))
import os

os.makedirs('../vectors/clean_for_pub/vocab', exist_ok=True)

save_files(yrs, loc, '../vectors/clean_for_pub/', 'sgns')
