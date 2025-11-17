import numpy as np
import os
import csv
import re

def load_vectors(filename):
    vectors = {}
    # 读的时候不用 newline=''
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            if not row:
                continue
            # 只保留字母的小写形式
            word = re.sub('[^a-z]+', '', row[0].strip().lower())
            if len(word) < 2:
                continue
            # 有些行末尾会多空字符串，这里过滤一下
            vec = [float(x) for x in row[1:] if len(x) > 0]
            vectors[word] = vec
    return vectors

def find_vector_norms(vectors):
    norms = [np.linalg.norm(vectors[word]) for word in vectors]
    return np.mean(norms), np.var(norms), np.median(norms)

def print_sizes(folder='../vectors/normalized_clean/'):
    # 这些文件名得跟你前面导出来的保持一致
    filenames_sgns = [folder + 'vectors_sgns{}.txt'.format(x) for x in range(1910, 2000, 10)]
    filenames_svd = [folder + 'vectors_svd{}.txt'.format(x) for x in range(1910, 2000, 10)]
    filenames_nyt = [folder + 'vectors{}-{}.txt'.format(x, x + 5) for x in range(1987, 2000, 1)]
    filenames_coha = [folder + 'vectorscoha{}-{}.txt'.format(x, x + 20) for x in range(1910, 2000, 10)]

    filenames_combined = [
        filenames_nyt,
        filenames_sgns,
        filenames_svd,
        [folder + 'vectorswikipedia.txt'],
        [folder + 'vectorsGoogleNews_exactclean.txt']
    ]

    for names in filenames_combined:
        for name in names:
            if not os.path.exists(name):
                print(f"{name} NOT FOUND, skip")
                continue
            stats = find_vector_norms(load_vectors(name))
            print(name, stats)

def normalize(filename, filename_output):
    countnorm0 = 0
    countnormal = 0

    # 确保输出目录存在
    os.makedirs(os.path.dirname(filename_output), exist_ok=True)

    # 写 csv 要 newline=''，不然 Windows 会多空行
    with open(filename_output, 'w', newline='', encoding='utf-8') as fo:
        writer = csv.writer(fo, delimiter=' ')
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                if not row:
                    continue
                # 拷贝一份要写出去的行
                rowout = list(row)
                word = re.sub('[^a-z]+', '', row[0].strip().lower())
                rowout[0] = word
                if len(word) < 2:
                    continue

                # 计算原始向量的范数
                vec = [float(x) for x in row[1:] if len(x) > 0]
                norm = np.linalg.norm(vec)

                if norm < 1e-2:
                    countnorm0 += 1
                else:
                    countnormal += 1
                    # 逐个除以范数
                    for i in range(1, len(rowout)):
                        if len(rowout[i]) > 0:
                            rowout[i] = float(rowout[i]) / norm
                    writer.writerow(rowout)

    print(countnorm0, countnormal)

if __name__ == "__main__":
    # 改成你上个脚本输出的路径
    os.makedirs('../vectors/normalized_clean/', exist_ok=True)
    folder = '../vectors/clean_for_pub/'
    filenames_sgns = [folder + 'vectors_sgns{}.txt'.format(x) for x in range(1910, 2000, 10)]

    for name in filenames_sgns:
        filename_output = name.replace('clean_for_pub/', 'normalized_clean/')
        print(name, '->', filename_output)
        normalize(name, filename_output)


