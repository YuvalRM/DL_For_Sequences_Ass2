import numpy as np


def sim(u, v):
    uv = np.dot(u, v)
    vv = np.dot(v, v)
    uu = np.dot(u, u)
    return uv / (np.sqrt(vv) * np.sqrt(uu))


def most_similar(word,k):
    print()

if __name__ == '__main__':
    vecs = np.loadtxt("vectors_file_name")