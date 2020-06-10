from othertools import *
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif
from XTREE import XTREE

from sklearn.preprocessing import MinMaxScaler

def RandomWalk(data_row):
    tem = data_row.copy()
    result =  [[ 0 for m in range(2)] for n in range(20)]
    for j in range(0,len(tem)):
        if np.random.rand(1)[0]>0.5:
            num1 = np.random.rand(1)[0]
            num2 = np.random.rand(1)[0]
            if num1<=num2:
                result[j][0],result[j][1] = num1,num2
            else:
                result[j][0],result[j][1] = num2,num1
            tem[j]=(num1+num2)/2
        else:
            result[j][0],result[j][1] = tem[j],tem[j]
    return tem,result

def RW(name, par, explainer=None, smote=False, small=0.05, act=False):
    files = [name[0], name[1], name[2]]
    freq = [0] * 20
    deltas = []
    for j in range(0, len(files) - 2):
        df1 = prepareData(files[j])
        df2 = prepareData(files[j + 1])
        for i in range(1, 21):
            col1 = df1.iloc[:, i]
            col2 = df2.iloc[:, i]
            deltas.append(hedge(col1, col2))
    #             if not (hedge(col1,col2,small)):
    #                 freq[i-1]+=1
    deltas = sorted(range(len(deltas)), key=lambda k: deltas[k], reverse=True)
    #     changed = dict()
    #     for i in range(20):
    #         freq[i] = 100*freq[i]/(len(files)-1)
    #         changed.update({df1.columns[i]:freq[i]})
    #     changed = list(changed.values())
    #     actionable = []
    #     for each in changed:
    #         actionable.append(1) if each!=0 else actionable.append(0)
    actionable = []
    for i in range(0, len(deltas)):
        if i in deltas[0:5]:
            actionable.append(1)
        else:
            actionable.append(0)
    print(actionable)
    df1 = prepareData(name[0])
    df2 = prepareData(name[1])
    df3 = prepareData(name[2])
    bug1 = bugs(name[0])
    bug2 = bugs(name[1])
    bug3 = bugs(name[2])
    df11 = df1.iloc[:, 1:]
    df22 = df2.iloc[:, 1:]
    df33 = df3.iloc[:, 1:]

    df1n = norm(df11, df11)
    df2n = norm(df22, df22)
    df3n = norm(df22, df33)

    X_train1 = df1n.iloc[:, :-1]
    y_train1 = df1n.iloc[:, -1]
    X_test1 = df2n.iloc[:, :-1]
    y_test1 = df2n.iloc[:, -1]
    X_test2 = df3n.iloc[:, :-1]
    y_test2 = df3n.iloc[:, -1]

    score = []
    bugchange = []
    size = []
    score2 = []
    para = 20
    clf1 = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=10, n_jobs=-1)
    #     clf1 =  MLPClassifier(hidden_layer_sizes=[10],max_iter=2000,early_stopping=False,learning_rate='adaptive')
    if smote:
        sm = SMOTE()
        X_train1_s, y_train1_s = sm.fit_resample(X_train1, y_train1)
        clf1.fit(X_train1_s, y_train1_s)
    else:
        clf1.fit(X_train1, y_train1)
    for i in range(0, len(y_test1)):
        for j in range(0, len(y_test2)):
            actual = X_test2.values[j]
            if df3.iloc[j, 0] == df2.iloc[i, 0] and y_test1[i] != 0:
                print('df2', i, 'df3', j)
                temp = X_test1.values[i].copy()
                tem, plan = RandomWalk(temp)
                score.append(overlap(X_test1.values[i], plan, actual))
                score2.append(overlap(X_test1.values[i], plan, X_test1.values[i]))
                size.append(size_interval(plan))
                bugchange.append(bug3[j] - bug2[i])  # negative if reduced #bugs, positive if added
                break
    print(name[0])
    print('>>>')
    print('>>>')
    print('>>>')
    return score, bugchange, size, score2

def planner(name, par, explainer=None, smote=False, small=0.05, act=False):
    # LIME: act = False
    # TimeLIME: act = True
    files = [name[0], name[1], name[2]]
    freq = [0] * 20
    deltas = []
    for j in range(0, len(files) - 2):
        df1 = prepareData(files[j])
        df2 = prepareData(files[j + 1])
        for i in range(1, 21):
            col1 = df1.iloc[:, i]
            col2 = df2.iloc[:, i]
            deltas.append(hedge(col1, col2))
    deltas = sorted(range(len(deltas)), key=lambda k: deltas[k], reverse=True)

    actionable = []
    for i in range(0, len(deltas)):
        if i in deltas[0:5]:
            actionable.append(1)
        else:
            actionable.append(0)
    print(actionable)
    df1 = prepareData(name[0])
    df2 = prepareData(name[1])
    df3 = prepareData(name[2])
    bug1 = bugs(name[0])
    bug2 = bugs(name[1])
    bug3 = bugs(name[2])
    df11 = df1.iloc[:, 1:]
    df22 = df2.iloc[:, 1:]
    df33 = df3.iloc[:, 1:]

    df1n = norm(df11, df11)
    df2n = norm(df22, df22)
    df3n = norm(df22, df33)

    X_train1 = df1n.iloc[:, :-1]
    y_train1 = df1n.iloc[:, -1]
    X_test1 = df2n.iloc[:, :-1]
    y_test1 = df2n.iloc[:, -1]
    X_test2 = df3n.iloc[:, :-1]
    y_test2 = df3n.iloc[:, -1]

    score = []
    bugchange = []
    size = []
    score2 = []
    par = 20
    clf1 = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=10, n_jobs=-1)
    #     clf1 =  MLPClassifier(hidden_layer_sizes=[10],max_iter=2000,early_stopping=False,learning_rate='adaptive')
    if smote:
        sm = SMOTE()
        X_train1_s, y_train1_s = sm.fit_resample(X_train1, y_train1)
        clf1.fit(X_train1_s, y_train1_s)
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train1_s, training_labels=y_train1_s,
                                                           feature_names=df11.columns,
                                                           discretizer='entropy', feature_selection='lasso_path',
                                                           mode='classification')
    else:
        clf1.fit(X_train1, y_train1)
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train1.values, training_labels=y_train1,
                                                           feature_names=df11.columns,
                                                           discretizer='entropy', feature_selection='lasso_path',
                                                           mode='classification')
    for i in range(0, len(y_test1)):
        for j in range(0, len(y_test2)):
            actual = X_test2.values[j]
            if df3.iloc[j, 0] == df2.iloc[i, 0] and y_test1[i] != 0:
                print('df2', i, 'df3', j)
                if clf1.predict([X_test1.values[i]]) == 0:
                    ins = explainer.explain_instance(data_row=X_test1.values[i], predict_fn=clf1.predict_proba,
                                                     num_features=20, num_samples=5000)
                    ind = ins.local_exp[1]
                    temp = X_test1.values[i].copy()
                    tem, plan, rej = flip(temp, ins.as_list(label=1), ind, clf1, df1n.columns, 0)
                    score.append(overlap(X_test1.values[i], plan, actual))
                    score2.append(overlap(X_test1.values[i], plan, X_test1.values[i]))
                    size.append(size_interval(plan))
                    bugchange.append(bug3[j] - bug2[i])  # negative if reduced #bugs, positive if added
                else:
                    ins = explainer.explain_instance(data_row=X_test1.values[i], predict_fn=clf1.predict_proba,
                                                     num_features=20, num_samples=5000)
                    ind = ins.local_exp[1]
                    temp = X_test1.values[i].copy()
                    if act:
                        tem, plan, rej = flip(temp, ins.as_list(label=1), ind, clf1, df1n.columns, par,
                                              actionable=actionable)
                    else:
                        tem, plan, rej = flip(temp, ins.as_list(label=1), ind, clf1, df1n.columns, par, actionable=None)
                    #                     if (tem != X_test1.values[i]).any():
                    if True:
                        score.append(overlap(X_test1.values[i], plan, actual))
                        size.append(size_interval(plan))
                        score2.append(overlap(X_test1.values[i], plan, X_test1.values[i]))
                        bugchange.append(bug3[j] - bug2[i])  # negative if reduced #bugs, positive if added
                break
    #     print("Runtime:",time.time()-start_time)
    print(name[0], par)
    print('>>>')
    print('>>>')
    print('>>>')
    return score, bugchange, size, score2

def _ent_weight(X, scale):
    try:
        loc = X["loc"].values  # LOC is the 10th index position.
    except KeyError:
        try:
            loc = X["$WCHU_numberOfLinesOfCode"].values
        except KeyError:
            loc = X["$CountLineCode"]

    return X.multiply(loc, axis="index") / scale

def alves(train, X_test, y_test, thresh=0.95):

    #     train.loc[train[train.columns[-1]] == 1, train.columns[-1]] = True
    #     train.loc[train[train.columns[-1]] == 0, train.columns[-1]] = False
    metrics = [met[1:] for met in train[train.columns[:-1]]]

    X = train  # Independent Features (CK-Metrics)
    changes = []

    """
    As weight we will consider
    the source lines of code (LOC) of the entity.
    """

    loc_key = "loc"
    tot_loc = train.sum()["loc"]
    X = _ent_weight(X, scale=tot_loc)

    """
    Divide the entity weight by the sum of all weights of the same system.
    """
    denom = pd.DataFrame(X).sum().values
    norm_sum = pd.DataFrame(pd.DataFrame(X).values / denom, columns=X.columns)

    """
    Find Thresholds
    """
    #     y = train[train.columns[-1]]  # Dependent Feature (Bugs)
    #     pVal = f_classif(X, y)[1]  # P-Values
    cutoff = []

    def cumsum(vals):
        return [sum(vals[:i]) for i, __ in enumerate(vals)]

    def point(array, thresh):
        for idx, val in enumerate(array):
            if val > thresh:
                return idx

    for idx in range(len(train.columns[:-1])):
        # Setup Cumulative Dist. Func.
        name = train.columns[idx]
        loc = train[loc_key].values
        vals = norm_sum[name].values
        sorted_ids = np.argsort(vals)
        cumulative = [sum(vals[:i]) for i, __ in enumerate(sorted(vals))]
        cutpoint = point(cumulative, thresh)
        cutoff.append(vals[sorted_ids[cutpoint]] * tot_loc / loc[sorted_ids[cutpoint]] * denom[idx])

    """
    Apply Plans Sequentially
    """

    modified = []
    for n in range(X_test.shape[0]):
        new_row = apply2(cutoff, X_test.iloc[n].values.tolist())
        modified.append(new_row)
    return pd.DataFrame(modified, columns=X_test.columns)

def runalves(name, thresh=0.7):
    df1 = prepareData(name[0])
    df2 = prepareData(name[1])
    df3 = prepareData(name[2])
    bug1 = bugs(name[0])
    bug2 = bugs(name[1])
    bug3 = bugs(name[2])
    df11 = df1.iloc[:, 1:]
    df22 = df2.iloc[:, 1:]
    df33 = df3.iloc[:, 1:]

    df1n = norm(df11, df11)
    df2n = norm(df22, df22)
    df3n = norm(df22, df33)

    X_train1 = df1n.iloc[:, :-1]
    y_train1 = df1n.iloc[:, -1]
    X_test1 = df2n.iloc[:, :-1]
    y_test1 = df2n.iloc[:, -1]
    X_test2 = df3n.iloc[:, :-1]
    y_test2 = df3n.iloc[:, -1]

    plans = alves(X_train1, X_test1, y_test1, thresh=thresh)
    score = []
    score2 = []
    bugchange = []
    size = []
    for i in range(0, len(y_test1)):
        for j in range(0, len(y_test2)):
            actual = X_test2.values[j]
            if df3.iloc[j, 0] == df2.iloc[i, 0] and y_test1[i] != 0:
                print('df2', i, 'df3', j)
                plan = plans.iloc[i, :].values
                #                 print("plan",plan)
                #                 print('actual',actual)
                #                 print('id1',plan[0][0])
                score.append(overlap(X_test1.values[i], plan, actual))
                score2.append(overlap(X_test1.values[i], plan, X_test1.values[i]))
                bugchange.append(bug3[j] - bug2[i])  # negative if reduced #bugs, positive if added
                size.append(size_interval(plan))
                break
    print(name[0])
    print('>>>')
    print('>>>')
    print('>>>')
    return score, bugchange, size, score2

def VARL(coef, inter, p0=0.05):
    """
    :param coef: Slope of   (Y=aX+b)
    :param inter: Intercept (Y=aX+b)
    :param p0: Confidence Interval. Default p=0.05 (95%)
    :return: VARL threshold

              1   /     /  p0   \             \
    VARL = ----- | log | ------ | - intercept |
           slope \     \ 1 - p0 /             /

    """
    return (np.log(p0 / (1 - p0)) - inter) / coef

def shatnawi(X_train, y_train, X_test, y_test, p):
    "Compute Thresholds"

    changed = []
    metrics = [str[1:] for str in X_train[X_train.columns[:]]]
    inter = []
    coef = []
    pVal = []
    for i in range(len(X_train.columns)):
        X = pd.DataFrame(X_train.iloc[:, i])  # Independent Features (CK-Metrics)
        y = y_train  # Dependent Feature (Bugs)
        ubr = LogisticRegression(solver='lbfgs')  # Init LogisticRegressor
        ubr.fit(X, y)  # Fit Logit curve
        inter.append(ubr.intercept_[0])  # Intercepts
        coef.append(ubr.coef_[0])  # Slopes
        pVal.append(f_classif(X, y)[1])  # P-Values
    changes = len(metrics) * [-1]
    "Find Thresholds using VARL"
    for Coeff, P_Val, Inter, idx in zip(coef, pVal, inter,
                                        range(len(metrics))):  # range(len(metrics)):
        thresh = VARL(Coeff, Inter, p0=p)  # default VARL p0=0.05 (95% CI)
        if P_Val < 0.05:
            changes[idx] = thresh
    """
    Apply Plans Sequentially
    """
    modified = []
    for n in range(X_test.shape[0]):
        new_row = apply3(changes, X_test.iloc[n, :].values.tolist())
        modified.append(new_row)
    return pd.DataFrame(modified, columns=X_test.columns)

def runshat(name, p=0.05):
    df1 = prepareData(name[0])
    df2 = prepareData(name[1])
    df3 = prepareData(name[2])
    bug1 = bugs(name[0])
    bug2 = bugs(name[1])
    bug3 = bugs(name[2])
    df11 = df1.iloc[:, 1:]
    df22 = df2.iloc[:, 1:]
    df33 = df3.iloc[:, 1:]

    df1n = norm(df11, df11)
    df2n = norm(df22, df22)
    df3n = norm(df22, df33)

    X_train1 = df1n.iloc[:, :-1]
    y_train1 = df1n.iloc[:, -1]
    X_test1 = df2n.iloc[:, :-1]
    y_test1 = df2n.iloc[:, -1]
    X_test2 = df3n.iloc[:, :-1]
    y_test2 = df3n.iloc[:, -1]

    plans = shatnawi(X_train1, y_train1, X_test1, y_test1, p=p)
    score = []
    score2 = []
    bugchange = []
    size = []
    for i in range(0, len(y_test1)):
        for j in range(0, len(y_test2)):
            actual = X_test2.values[j]
            if df3.iloc[j, 0] == df2.iloc[i, 0] and y_test1[i] != 0:
                print('df2', i, 'df3', j)
                plan = plans.iloc[i, :].values
                #                 print("plan",plan)
                #                 print('actual',actual)
                #                 print('id1',plan[0][0])
                score.append(overlap(X_test1.values[i], plan, actual))
                score2.append(overlap(X_test1.values[i], plan, X_test1.values[i]))
                bugchange.append(bug3[j] - bug2[i])  # negative if reduced #bugs, positive if added
                size.append(size_interval(plan))
                break
    print(name[0])
    print('>>>')
    print('>>>')
    print('>>>')
    return score, bugchange, size, score2


def get_percentiles(df):
    percentile_array = []
    q = dict()
    for i in np.arange(0, 100, 1):
        for col in df.columns:
            try:
                q.update({col: np.percentile(df[col].values, q=i)})
            except:
                pass

        elements = dict()
        for col in df.columns:
            try:
                elements.update({col: df.loc[df[col] >= q[col]].median()[col]})
            except:
                pass

        percentile_array.append(elements)

    return percentile_array

def oliveira(train, test):
    """
    Implements shatnavi's threshold based planner.
    :param train:
    :param test:
    :param rftrain:
    :param tunings:
    :param verbose:
    :return:
    """
    "Helper Functions"

    def compliance_rate(k, train_columns):
        return len([t for t in train_columns if t <= k]) / len(train_columns)

    def penalty_1(p, k, Min, compliance):

        comply = Min - compliance
        if comply >= 0:
            return (Min - compliance) / Min
        else:
            return 0

    def penalty_2(k, Med):
        if k > Med:
            return (k - Med) / Med
        else:
            return 0

    "Compute Thresholds"

    #     if isinstance(test, list):
    #         test = list2dataframe(test)

    #     if isinstance(test, str):
    #         test = list2dataframe([test])

    #     if isinstance(train, list):
    #         train = list2dataframe(train)

    lo, hi = train.min(), train.max()
    quantile_array = get_percentiles(train)
    changes = []

    pk_best = dict()

    for metric in train.columns[:]:
        min_comply = 10e32
        vals = np.empty([10, 100])
        for p_id, p in enumerate(np.arange(0, 100, 10)):
            p = p / 100
            for k_id, k in enumerate(np.linspace(lo[metric], hi[metric], 100)):
                try:
                    med = quantile_array[90][metric]
                    compliance = compliance_rate(k, train[metric])
                    penalty1 = penalty_1(p, k, compliance=compliance, Min=0.9)
                    penalty2 = penalty_2(k, med)
                    comply_rate_penalty = penalty1 + penalty2
                    vals[p_id, k_id] = comply_rate_penalty

                    if (comply_rate_penalty < min_comply) or (
                            comply_rate_penalty == min_comply and p >= pk_best[metric][0] and k <= pk_best[metric][1]):
                        min_comply = comply_rate_penalty
                        try:
                            pk_best[metric] = (p, k)
                            print('p k best', p, k)
                        except KeyError:
                            pk_best.update({metric: (p, k)})
                except:
                    pk_best.update({metric: (p, None)})
        print('comply', metric, min_comply)

    """
    Apply Plans Sequentially
    """

    modified = []
    for n in range(test.shape[0]):
        new_row = apply4(test.iloc[n].values.tolist(), test.columns, pk_best)
        modified.append(new_row)
    return pd.DataFrame(modified, columns=test.columns)

def runolive(name):
    df1 = prepareData(name[0])
    df2 = prepareData(name[1])
    df3 = prepareData(name[2])
    bug1 = bugs(name[0])
    bug2 = bugs(name[1])
    bug3 = bugs(name[2])
    df11 = df1.iloc[:, 1:]
    df22 = df2.iloc[:, 1:]
    df33 = df3.iloc[:, 1:]

    df1n = norm(df11, df11)
    df2n = norm(df22, df22)
    df3n = norm(df22, df33)

    X_train1 = df1n.iloc[:, :-1]
    y_train1 = df1n.iloc[:, -1]
    X_test1 = df2n.iloc[:, :-1]
    y_test1 = df2n.iloc[:, -1]
    X_test2 = df3n.iloc[:, :-1]
    y_test2 = df3n.iloc[:, -1]

    plans = oliveira(X_train1, X_test1)
    score = []
    bugchange = []
    size = []
    score2 = []
    for i in range(0, len(y_test1)):
        for j in range(0, len(y_test2)):
            actual = X_test2.values[j]
            if df3.iloc[j, 0] == df2.iloc[i, 0] and y_test1[i] != 0:
                print('df2', i, 'df3', j)
                plan = plans.iloc[i, :].values
                #                 print("plan",plan)
                #                 print('actual',actual)
                #                 print('id1',plan[0][0])
                score.append(overlap(X_test1.values[i], plan, actual))
                bugchange.append(bug3[j] - bug2[i])  # negative if reduced #bugs, positive if added
                size.append(size_interval(plan))
                score2.append(overlap(X_test1.values[i], plan, X_test1.values[i]))
                break
    print(name[0])
    print('>>>')
    print('>>>')
    print('>>>')
    return score, bugchange, size, score2

def xtree(name):
    df1 = prepareData(name[0])
    df2 = prepareData(name[1])
    df3 = prepareData(name[2])
    bug1 = bugs(name[0])
    bug2 = bugs(name[1])
    bug3 = bugs(name[2])
    df11 = df1.iloc[:, 1:]
    df22 = df2.iloc[:, 1:]
    df33 = df3.iloc[:, 1:]
    df1n = norm(df11,df11)
    df2n = norm(df11,df22)
    df3n = norm(df11,df33)
    X_train1 = df1n.iloc[:, :-1]
    y_train1 = df1n.iloc[:, -1]
    X_test1 = df2n.iloc[:, :-1]
    y_test1 = df2n.iloc[:, -1]
    X_test2 = df3n.iloc[:, :-1]
    y_test2 = df3n.iloc[:, -1]
    X_test = pd.concat([X_test1, y_test1], axis=1, ignore_index=True)
    X_test.columns = df1.columns[1:]

    xtree_arplan = XTREE(strategy="closest",alpha=0.9,support_min=int(X_train1.shape[0]/20))
    xtree_arplan = xtree_arplan.fit(X_train1)
    patched_xtree = xtree_arplan.predict(X_test)
    print(patched_xtree.shape[0], X_test1.shape[0])
    XTREE.pretty_print(xtree_arplan)
    overlap_scores = []
    bcs = []
    size=[]
    score2=[]
    for i in range(0, X_test1.shape[0]):
        for j in range(0, X_test2.shape[0]):
            if df3.iloc[j, 0] == df2.iloc[i, 0] and y_test1[i] != 0:
                plan = patched_xtree.iloc[i, :-1]
                # print('**********')
                # print('plan:',plan.values)
                # print('ori:',X_test1.iloc[i,:].values)
                # print('**********')
                actual = X_test2.iloc[j, :]
                overlap_scores.append(overlap1(plan, plan, actual))
                bcs.append(bug3[j]-bug2[i])
                score2.append(overlap(X_test1.values[i], plan, X_test1.values[i]))
                size.append(size_interval(plan))
                break
    return overlap_scores,bcs,size,score2

