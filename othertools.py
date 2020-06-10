import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from sklearn.ensemble import RandomForestClassifier
import lime.lime_tabular
from imblearn.over_sampling import SMOTE
from scipy import stats
import collections
## This file contains all tools needed to run planners.

def prepareData(fname):
    # Read csv file into dataframe
    # print(os.path.join())
    file = os.path.join("Data",fname)
    df = pd.read_csv(file,sep=',')
    for i in range(0,df.shape[0]):
        if df.iloc[i,-1] >0:
            df.iloc[i,-1]  = 1
        else:
            df.iloc[i,-1]  = 0
    return df

def bugs(fname):
    # return the number of bugs in each row
    file = os.path.join("Data", fname)
    df = pd.read_csv(file,sep=',')
    return df.iloc[:,-1]

def translate1(sentence, name):
    # translate LIME explanations in to 2 values, representing the interval
    lst = sentence.strip().split(name)
    left, right = 0, 0
    if lst[0] == '':
        del lst[0]
    if len(lst) == 2:
        if '<=' in lst[1]:
            aa = lst[1].strip(' <=')
            right = float(aa)
        elif '<' in lst[1]:
            aa = lst[1].strip(' <')
            right = float(aa)
        if '<=' in lst[0]:
            aa = lst[0].strip(' <=')
            left = float(aa)
        elif '<' in lst[0]:
            aa = lst[0].strip(' <')
            left = float(aa)
    else:
        if '<=' in lst[0]:
            aa = lst[0].strip(' <=')
            right = float(aa)
            left = 0
        elif '<' in lst[0]:
            aa = lst[0].strip(' <')
            right = float(aa)
            left = 0
        if '>=' in lst[0]:
            aa = lst[0].strip(' >=')
            left = float(aa)
            right = 1
        elif '>' in lst[0]:
            aa = lst[0].strip(' >')
            left = float(aa)
            right = 1
    return left, right


def translate(sentence, name):
    # not used
    flag = 0
    threshold = 0
    lst = sentence.strip().split(name)
    #     print('LST',lst)
    if lst[0] == '':
        del lst[0]
    if len(lst) == 2:
        if '<=' in lst[1]:
            flag = 1
            aa = lst[1].strip(' <=')
            threshold1 = float(aa)
        elif '<' in lst[1]:
            flag = 1
            aa = lst[1].strip(' <')
            threshold1 = float(aa)
        if '<=' in lst[0]:
            flag = -1
            aa = lst[0].strip(' <=')
            threshold0 = float(aa)
        elif '<' in lst[0]:
            flag = -1
            aa = lst[0].strip(' <')
            threshold0 = float(aa)
        if threshold0 == 0:
            result = threshold1
            flag = 1
        elif (1 - threshold1) >= (threshold0 - 0):
            result = threshold1
            flag = 1
        else:
            result = threshold0
            flag = -1
    else:
        if '<=' in lst[0]:
            flag = 1
            aa = lst[0].strip(' <=')
            threshold = float(aa)
        elif '<' in lst[0]:
            flag = 1
            aa = lst[0].strip(' <')
            threshold = float(aa)
        if '>=' in lst[0]:
            flag = -1
            aa = lst[0].strip(' >=')
            threshold = float(aa)
        elif '>' in lst[0]:
            flag = -1
            aa = lst[0].strip(' >')
            threshold = float(aa)
        result = threshold
    return flag, result


def flip(data_row, local_exp, ind, clf, cols, n_feature=3, actionable=None):
    counter = 0
    rejected = 0
    cache = []
    trans = []
    # Store feature index in cache.
    cnt = []
    for i in range(0, len(local_exp)):
        cache.append(ind[i])
        trans.append(local_exp[i])
        if ind[i][1] > 0:
            cnt.append(i)
    tem = data_row.copy()
    result = [[0 for m in range(2)] for n in range(20)]
    for j in range(0, len(local_exp)):
        act = True
        if actionable:
            if actionable[j] == 0:
                act = False
        l, r = translate1(trans[j][0], cols[cache[j][0]])
        if j in cnt and counter < n_feature and act:
            # features needed to be altered
            if (l + r) / 2 < 0.5:
                if r + r - l <= 1:
                    result[cache[j][0]][0], result[cache[j][0]][1] = r, r + (r - l)
                else:
                    result[cache[j][0]][0], result[cache[j][0]][1] = r, 1
            else:
                if l - (r - l) >= 0:
                    result[cache[j][0]][0], result[cache[j][0]][1] = l - (r - l), l
                else:
                    result[cache[j][0]][0], result[cache[j][0]][1] = 0, l
            tem[cache[j][0]] = (result[cache[j][0]][0] + result[cache[j][0]][1]) / 2
            counter += 1
        else:
            if act == False:
                rejected += 1
            l, r = translate1(trans[j][0], cols[cache[j][0]])
            result[cache[j][0]][0], result[cache[j][0]][1] = l, r
    return tem, result, rejected

def hedge(arr1,arr2):
    # returns a value, larger means more changes
    s1,s2 = np.std(arr1),np.std(arr2)
    m1,m2 = np.mean(arr1),np.mean(arr2)
    n1,n2 = len(arr1),len(arr2)
    num = (n1-1)*s1**2 + (n2-1)*s2**2
    denom = n1+n2-1-1
    sp = (num/denom)**.5
    delta = np.abs(m1-m2)/sp
    c = 1-3/(4*(denom)-1)
    return delta*c


def norm (df1,df2):
    # min-max scale the dataset
    X1 = df1.iloc[:,:-1].values
    mm = MinMaxScaler()
    mm.fit(X1)
    X2 = df2.iloc[:,:-1].values
    X2 = mm.transform(X2)
    df2 = df2.copy()
    df2.iloc[:,:-1] = X2
    return df2


def overlap(ori,plan,actual): # Jaccard similarity function
    cnt = 20
    right = 0
    # print(plan)
    for i in range(0,len(plan)):
        if isinstance(plan[i], float):
            if np.round(actual[i],4)== np.round(plan[i],4):
                right+=1
        else:
            if actual[i]>=0 and actual[i]<=1:
                if actual[i]>=plan[i][0] and actual[i]<=plan[i][1]:
                    right+=1
            elif actual[i]>1:
                if plan[i][1]>=1:
                    right+=1
            else:
                if plan[i][0]<=0:
                    right+=1
    return right/cnt

def overlap1(ori,plan,actual): # Jaccard similarity function
    cnt = 20
    right = 0
    # print(plan)
    for i in range(0,len(plan)):
        if isinstance(plan[i], list):
            if actual[i]>=plan[i][0] and actual[i]<=plan[i][1]:
                right+=1
        else:
            if actual[i]==plan[i]:
                right+=1
    return right/cnt

def size_interval(plan):
    out=[]
    for i in range(len(plan)):
        if not isinstance(plan[i],float):
            out.append(plan[i][1]-plan[i][0])
        else:
            out.append(0)
    return out

def apply3(changes, row):
    new_row = row
    for idx, thres in enumerate(changes):
        if thres > 0:
            try:
                if new_row[idx] > thres:
                    new_row[idx] = (0, thres)
            except:
                pass
    return new_row

def apply2(changes, row):
    new_row = row
    for idx, thres in enumerate(changes):
        if thres is not None:
            try:
                if new_row[idx] > thres:
                    new_row[idx] = (0, thres)
            except:
                pass
    return new_row

def apply4(row, cols, pk_best):
    newRow = row
    for idx, col in enumerate(cols):
        try:
            thres = pk_best[col][1]
            proba = pk_best[col][0]
            if thres is not None:
                if newRow[idx] > thres:
                    print("Yes", thres, proba)
                    newRow[idx] = (0, thres)
        #                     if random(0, 100) < proba else \newRow[idx]
        except:
            pass
    return newRow

def cf (lst):
    res=[]
    for i in range(len(lst)):
        res.append([float(each) for each in lst[i].strip('[]').split(',')])
    return res

def readfile(fname):
    file = pd.read_csv(fname, sep=',')
    file.drop(columns=file.columns[0], inplace=True)
    result = []
    N = file.shape[1]
    for i in range(N):
        temp = []
        for k in range(file.shape[1]):

            if pd.isnull(file.iloc[i, k]):
                continue
            else:
                temp.append(file.iloc[i, k])
        result.append(temp)
    return result