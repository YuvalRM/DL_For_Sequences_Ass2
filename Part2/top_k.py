import numpy as np


def sim(u, v):
    uv = np.dot(u, v)
    vv = np.dot(v, v)
    uu = np.dot(u, u)
    return uv / (np.sqrt(vv) * np.sqrt(uu))


def most_similar(word, k):
    vecs = np.loadtxt("../wordVectors.txt")
    f = open("../vocab.txt", "r")
    vocabulary = f.readlines()
    vocabulary = [word.strip() for word in vocabulary]
    f.close()
    voc_vecs = zip(vocabulary, vecs)
    word_dic = {word: vec for word, vec in voc_vecs}
    vec=word_dic[word]

    closest = [['' for i in range(k)], [-np.inf for i in range(k)]]
    for w, v in word_dic.items():
        if w==word:
            continue
        dist = sim(vec, v)
        if dist > min(closest[1]):
            location=np.argmin(closest[1])
            closest[0][location] = w
            closest[1][location] = dist



    print(word_dic[word])
    return closest[0]


if __name__ == '__main__':
    print(most_similar('dog', 5))
    print(most_similar('england', 5))
    print(most_similar('john', 5))
    print(most_similar('explode', 5))
    print(most_similar('office', 5))
