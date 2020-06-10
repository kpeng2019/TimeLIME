from planner import *
from othertools import *
import matplotlib.pyplot as plt


def main():
    scores_t = readfile('rq2_TimeLIME.csv')
    scores_f = readfile('rq2_LIME.csv')
    scores_x = readfile('rq2_XTREE.csv')
    scores_alve = readfile('rq2_Alves.csv')
    scores_shat = readfile('rq2_Shat.csv')
    scores_oliv = readfile('rq2_Oliv.csv')
    scores_rw = readfile('rq2_Random.csv')

    bcs_t = readfile('rq3_TimeLIME.csv')
    bcs_f = readfile('rq3_LIME.csv')
    bcs_x = readfile('rq3_XTREE.csv')
    bcs_alve = readfile('rq3_Alves.csv')
    bcs_shat = readfile('rq3_Shat.csv')
    bcs_oliv = readfile('rq3_Oliv.csv')
    bcs_rw = readfile('rq3_Random.csv')
    list1 = [scores_t,scores_f,scores_x,scores_alve,scores_shat,scores_oliv,scores_rw]
    list2 = [bcs_t,bcs_f,bcs_x,bcs_alve,bcs_shat,bcs_oliv,bcs_rw]
    names = ['TimeLIME','LIME','XTREE','Alves','Shatnawi','Oliveira','Random']
    results=[]
    for i in range(len(names)):
        scores = list1[i]
        bcs = list2[i]
        dummy = []
        N = len(scores)
        for i in range(0, len(scores)):
            temp = 0
            for j in range(0, len(scores[i])):
                temp -= (bcs[i][j] * scores[i][j])
            total = -np.sum(bcs[i])
            dummy.append(np.round(temp / total, 3))
        print(names[i],dummy)
        results.append(dummy)
    return results

if __name__ == "__main__":
    main()
