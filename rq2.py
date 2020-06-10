from planner import *
from othertools import *
import matplotlib.pyplot as plt
import scipy as sp

def main():
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
    scores_t = readfile('rq2_TimeLIME.csv')
    scores_f = readfile('rq2_LIME.csv')
    scores_x = readfile('rq2_XTREE.csv')
    scores_alve = readfile('rq2_Alves.csv')
    scores_shat = readfile('rq2_Shat.csv')
    scores_oliv = readfile('rq2_Oliv.csv')
    scores_rw = readfile('rq2_Random.csv')

    N = len(scores_t)
    for i in range(N):
        print()
        print(fnames[i][0])
        print('TimeLIME', sp.stats.iqr(scores_t[i]), np.median(scores_t[i]))
        print('LIME', sp.stats.iqr(scores_f[i]), np.median(scores_f[i]))
        print('XTREE', sp.stats.iqr(scores_x[i]), np.median(scores_x[i]))
        print('Alves', sp.stats.iqr(scores_alve[i]), np.median(scores_alve[i]))
        print('Shatnawi', sp.stats.iqr(scores_shat[i]), np.median(scores_shat[i]))
        print('Oliveira', sp.stats.iqr(scores_oliv[i]), np.median(scores_oliv[i]))
        print('Random', sp.stats.iqr(scores_rw[i]), np.median(scores_rw[i]))
    return


if __name__ == "__main__":
    main()
