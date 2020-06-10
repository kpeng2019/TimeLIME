from planner import *
import matplotlib.pyplot as plt


def main():
    # TimeLIME planner
    paras = [True]
    explainer = None
    fnames = [['jedit-4.0.csv', 'jedit-4.1.csv', 'jedit-4.2.csv'],
              ['camel-1.0.csv', 'camel-1.2.csv', 'camel-1.4.csv'],
              ['camel-1.2.csv', 'camel-1.4.csv', 'camel-1.6.csv'],
              ['log4j-1.0.csv', 'log4j-1.1.csv', 'log4j-1.2.csv'],
              ['xalan-2.4.csv', 'xalan-2.5.csv', 'xalan-2.6.csv'],
              ['ant-1.5.csv', 'ant-1.6.csv', 'ant-1.7.csv'],
              ['velocity-1.4.csv', 'velocity-1.5.csv', 'velocity-1.6.csv'],
              ['poi-1.5.csv', 'poi-2.5.csv', 'poi-3.0.csv'],
              ['synapse-1.0.csv', 'synapse-1.1.csv', 'synapse-1.2.csv']
              ]
    scores_t, bcs_t = [], []
    size_t, score_2t = [], []
    for par in paras:
        for name in fnames:
            score, bc, size, score_2 = planner(name, 20, explainer, smote=True, small=.03, act=par)
            scores_t.append(score)
            bcs_t.append(bc)
            size_t.append(size)
            score_2t.append(score_2)

    # Classical LIME planner
    paras = [False]
    explainer = None
    scores_f, bcs_f = [], []
    size_f, score_2f = [], []
    for par in paras:
        for name in fnames:
            score, bc, size, score_2 = planner(name, 20, explainer, smote=True, small=.03, act=par)
            scores_f.append(score)
            bcs_f.append(bc)
            size_f.append(size)
            score_2f.append(score_2)

    # Random planner
    scores_rw, bcs_rw = [], []
    size_rw, score2_rw = [], []
    for name in fnames:
        score, bc, size, score_2 = RW(name, 20, explainer, smote=False, small=.03, act=False)
        scores_rw.append(score)
        bcs_rw.append(bc)
        size_rw.append(size)
        score2_rw.append(score_2)

    # Alves
    scores_alve, bcs_alve, sizes_alve, scores2_alve = [], [], [], []
    for name in fnames:
        score, bc, size, score2 = runalves(name, thresh=0.95)
        scores_alve.append(score)
        bcs_alve.append(bc)
        sizes_alve.append(size)
        scores2_alve.append(score2)

    # Shatnawi
    scores_shat, bcs_shat, sizes_shat, scores2_shat = [], [], [], []
    for name in fnames:
        score, bc, size, score2 = runshat(name, 0.5)
        scores_shat.append(score)
        bcs_shat.append(bc)
        sizes_shat.append(size)
        scores2_shat.append(score2)

    # Oliveira
    scores_oliv, bcs_oliv, sizes_oliv, scores2_oliv = [], [], [], []
    for name in fnames:
        score, bc, size, score2 = runolive(name)
        scores_oliv.append(score)
        bcs_oliv.append(bc)
        sizes_oliv.append(size)
        scores2_oliv.append(score2)

    # XTREE
    scores_x, bcs_x, sizes_x, scores2_x = [], [], [], []
    for par in paras:
        for name in fnames:
            score_x, bc_x, size_x, score2 = xtree(name)
            scores_x.append(score_x)
            bcs_x.append(bc_x)
            sizes_x.append(size_x)
            scores2_x.append(score2)

    pd.DataFrame(score_2t).to_csv("rq1_TimeLIME.csv")
    pd.DataFrame(score_2f).to_csv("rq1_LIME.csv")
    pd.DataFrame(scores2_x).to_csv("rq1_XTREE.csv")
    pd.DataFrame(scores2_alve).to_csv("rq1_Alves.csv")
    pd.DataFrame(scores2_oliv).to_csv("rq1_Oliv.csv")
    pd.DataFrame(scores2_shat).to_csv("rq1_Shat.csv")
    pd.DataFrame(score2_rw).to_csv("rq1_Random.csv")

    pd.DataFrame(scores_t).to_csv("rq2_TimeLIME.csv")
    pd.DataFrame(scores_f).to_csv("rq2_LIME.csv")
    pd.DataFrame(scores_x).to_csv("rq2_XTREE.csv")
    pd.DataFrame(scores_alve).to_csv("rq2_Alves.csv")
    pd.DataFrame(scores_oliv).to_csv("rq2_Oliv.csv")
    pd.DataFrame(scores_shat).to_csv("rq2_Shat.csv")
    pd.DataFrame(scores_rw).to_csv("rq2_Random.csv")

    pd.DataFrame(bcs_t).to_csv("rq3_TimeLIME.csv")
    pd.DataFrame(bcs_f).to_csv("rq3_LIME.csv")
    pd.DataFrame(bcs_x).to_csv("rq3_XTREE.csv")
    pd.DataFrame(bcs_alve).to_csv("rq3_Alves.csv")
    pd.DataFrame(bcs_oliv).to_csv("rq3_Oliv.csv")
    pd.DataFrame(bcs_shat).to_csv("rq3_Shat.csv")
    pd.DataFrame(bcs_rw).to_csv("rq3_Random.csv")



    plt.subplots(figsize=(7, 7))
    plt.rcParams.update({'font.size': 16})
    # ind=np.arange(10)
    N = len(scores2_x)
    width = 0.25
    dummy1,dummy2,dummy3,dummy4,dummy5,dummy6,dummy7 = [],[],[],[],[],[],[]
    for i in range(0,len(scores2_x)):
        dummy1.append(np.round(1-np.mean(score_2t[i]),3)*20)
        dummy2.append(np.round(1-np.mean(score_2f[i]),3)*20)
        dummy3.append(np.round(1-np.mean(scores2_x[i]),3)*20)
        dummy4.append(np.round(1-np.mean(scores2_alve[i]),3)*20)
        dummy5.append(np.round(1-np.mean(scores2_shat[i]),3)*20)
        dummy6.append(np.round(1-np.mean(scores2_oliv[i]),3)*20)
        dummy7.append(np.round(1-np.mean(score2_rw[i]),3)*20)
    plt.scatter(np.arange(N), dummy2, label='Classical LIME', s=100, marker='o')
    plt.scatter(np.arange(N), dummy3, label='XTREE', s=100, marker='o')
    plt.scatter(np.arange(N), dummy4, label='Alves', s=100, marker='o')
    plt.scatter(np.arange(N), dummy5, label='Shatnawi', s=100, marker='o')
    plt.scatter(np.arange(N), dummy6, label='Oliveira', s=100, marker='o')
    plt.scatter(np.arange(N), dummy7, label='RandomWalk', s=100, marker='v')
    plt.plot(np.arange(N), dummy1, label='TimeLIME', marker='^', markersize=10, color='#22406D')

    # plt.ylim(-11,130)
    plt.xticks(np.arange(N), ['jedit', 'camel1', 'camel2', 'log4j', 'xalan', 'ant', 'velocity', 'poi', 'synapse'])
    plt.yticks([0, 2, 4, 6, 8, 10, 12])
    plt.subplots_adjust(bottom=0.2, left=0, right=1.1)
    plt.grid(axis='y')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3)
    plt.savefig("rq1", dpi=200, bbox_inches='tight')
    plt.show()

    return
if __name__ == "__main__":
    main()
