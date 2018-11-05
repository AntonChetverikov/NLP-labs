# -*- coding: utf-8 -*-
# HMM Viterbi POS tagger

from conllu import parse_incr
from io import open

def HMM_Viterbi_POStagger(train_file, test_file):

    # Set of possible tag, K[-1] = K[0] = "*"
    def K(n):
        if n == -1 or n == 0:
            return ["*"]
        else:
            return tags

    # Trigram HMM parameters:

    def q(s, U, V):
        eps = 1e-7
        return tC.get((U, V, s), eps) / bC.get((U, V), eps)

    def e(x, s):
        eps = 1e-7
        return wtC.get((s, x), eps) / uC.get((s), eps)
    

    def Viterbi_algo(sentence):
        Pi = {(0, "*", "*"): 1}
        bp = {}
        n = len(sentence) - 2
        y = [""] * (n + 1)

        for k in range(1, n + 1):
            x_k = sentence[k]

            for U in K(k-1):
                for V in K(k):
                    W = K(k-2)
                    V_1 = map(lambda W_i:
                                Pi.get((k - 1, W_i, U)) *
                                q(V, W_i, U) *
                                e(x_k, V), W)
                    V_1 = list(V_1)
                    PiNew = max(V_1)
                    bpNew = W[V_1.index(PiNew)]
                    Pi.update({(k, U, V): PiNew})
                    bp.update({(k, U, V): bpNew})

        V_2 = {}
        for U in K(n - 1):
            for V in K(n):
                V_2.update({(n, U, V): Pi.get((n, U, V)) * q("STOP", U, V)})

        V_2max = max(list(V_2.values()))
        for (n, U, V), val in V_2.items():
            if val == V_2max:
                (y[n - 1], y[n]) = (U, V)

        for k in range(n - 2, 0, -1):
            y[k] = bp.get((k + 2, y[k + 1], y[k + 2]))

        return y[1:]

    
    # Corpus parsing
    train_data = parse_incr(open(train_file, "r", encoding = "utf-8"))
    
    wtC = {}    # Word-tag count
    uC = {}     # Unigram count
    bC = {}     # Bigram count
    tC = {}     # Trigram count

    # Making * for start of sentence and "STOP" for end of sentence
    for tokenlist in train_data:
        sentence = [["*", "*"]]
        for i in range(len(tokenlist)):
            sentence += [[tokenlist[i]["lemma"], tokenlist[i]["upostag"]]]        
        sentence += [["STOP","STOP"]]

    # Making uni-, bi-, trigrams counts and word-tag counts
        for i in range(0, len(sentence) - 2):
            tC_old = tC.get((sentence[i][1], sentence[i + 1][1], sentence[i + 2][1]), 0)
            tC.update({(sentence[i][1], sentence[i + 1][1], sentence[i + 2][1]): tC_old + 1 })

        for i in range(0, len(sentence) - 1):
            bC_old = bC.get( (sentence[i][1], sentence[i + 1][1]), 0)
            bC.update({(sentence[i][1], sentence[i + 1][1]): bC_old + 1})

        for i in range(0, len(sentence)):
            uC_old = uC.get((sentence[i][1]), 0)
            uC.update({(sentence[i][1]): uC_old + 1})

        for i in range(0, len(sentence)):
            wtC_old = wtC.get((sentence[i][1], sentence[i][0]), 0)
            wtC.update({(sentence[i][1], sentence[i][0]): wtC_old + 1})

    # Making tag list
    tags = list(uC.keys())
    tags.remove('*')
    tags.remove('STOP')
    
    test_data = parse_incr(open(test_file, "r", encoding = "utf-8"))
    test_tags = [] # Tags from test corpus
    predict_tags = [] # Tags from algorithm for test corpus

    for tokenlist in test_data:
        sentence = ["*"] + [tokenlist[i]["lemma"] for i in range(len(tokenlist)) ] + ["STOP"]
        test_tags.append([tokenlist[i]["upostag"] for i in range(len(tokenlist))])
        predict_tags.append(Viterbi_algo(sentence))
    
    return test_tags, predict_tags