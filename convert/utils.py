"""
Part of a [Recidivism Case Study](https://allendowney.github.io/RecidivismCaseStudy/)

by [Allen Downey](https://allendowney.com)

[Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def values(series):
    """Count the values and sort.

    series: pd.Series

    returns: series mapping from values to frequencies
    """
    return series.value_counts(dropna=False).sort_index()


def make_matrix(cp, threshold=4):
    """Make a confusion matrix.

    cp: DataFrame
    threshold: default is 4

    returns: DataFrame containing the confusion matrix
    """
    # recode the decile scores
    a = np.where(cp['decile_score'] > threshold,
                 'Positive',
                 'Negative')
    high_risk = pd.Series(a, name='Predicted')

    # recode the recidivism outcomes
    a = np.where(cp['two_year_recid'] == 1,
                 'Condition',
                 'No Condition')
    new_charge_2 = pd.Series(a, name='Actual')

    # cross tabulate
    matrix = pd.crosstab(high_risk, new_charge_2)

    # make sure all four elements are present
    index = ['Positive', 'Negative']
    columns = ['Condition', 'No Condition']
    matrix = matrix.reindex(index=index, columns=columns, fill_value=0)

    return matrix


def percent(x, y):
    """Compute the percentage `x/(x+y)*100`.
    """
    if x+y == 0:
        return np.nan
    return x / (x+y) * 100


def predictive_value(m):
    """Compute positive and negative predictive value.

    m: confusion matrix
    """
    tp, fp, fn, tn = m.to_numpy().flatten()
    ppv = percent(tp, fp)
    npv = percent(tn, fn)
    return ppv, npv


def sens_spec(m):
    """Compute sensitivity and specificity.

    m: confusion matrix
    """
    tp, fp, fn, tn = m.to_numpy().flatten()
    sens = percent(tp, fn)
    spec = percent(tn, fp)
    return sens, spec


def error_rates(m):
    """Compute false positive and false negative rate.

    m: confusion matrix
    """
    tp, fp, fn, tn = m.to_numpy().flatten()
    fpr = percent(fp, tn)
    fnr = percent(fn, tp)
    return fpr, fnr


def prevalence(df):
    """Compute prevalence.

    m: confusion matrix
    """
    tp, fp, fn, tn = df.to_numpy().flatten()
    prevalence = percent(tp+fn, tn+fp)
    return prevalence


def compute_metrics(m, name=''):
    """Compute all metrics.

    m: confusion matrix

    returns: DataFrame
    """
    fpr, fnr = error_rates(m)
    ppv, npv = predictive_value(m)
    prev = prevalence(m)

    index = ['FPR', 'FNR', 'PPV', 'NPV', 'Prevalence']
    df = pd.DataFrame(index=index, columns=['Percent'])
    df.Percent = fpr, fnr, ppv, npv, prev
    df.index.name = name
    return df


def calibration_curve(df):
    """Compute probability of recidivism by decile score.

    df: DataFrame

    returns: Series
    """
    grouped = df.groupby('decile_score')
    return grouped['two_year_recid'].mean()


def decorate(**options):
    """Decorate the current axes.
    
    Call decorate with keyword arguments like
    decorate(title='Title',
             xlabel='x',
             ylabel='y')
             
    The keyword arguments can be any of the axis properties
    https://matplotlib.org/api/axes_api.html
    """
    ax = plt.gca()
    ax.set(**options)
    
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels)

    plt.tight_layout()

def constant_predictive_value(ppv, npv, prev):
    """Make a confusion matrix with given metrics.

    ppv: positive predictive value
    npv: negative predictive value
    prev: prevalence

    returns: confusion matrix
    """
    ppv /= 100
    npv /= 100
    prev /= 100
    a = np.array((ppv*(npv + prev - 1)/(npv + ppv - 1),
                -(ppv - 1)*(npv + prev - 1)/(npv + ppv - 1),
                -(npv - 1)*(ppv - prev)/(npv + ppv - 1),
                npv*(ppv - prev)/(npv + ppv - 1)))

    return pd.DataFrame(a.reshape((2, 2)))


def run_cpv_model(cp):
    """Run the constant predictive value model.

    cp: DataFrame of COMPAS data

    returns: DataFrame with a row for each prevalence and
             a column for each metric (FPR, FNR)
    """
    matrix_all = make_matrix(cp)

    ppv, npv = predictive_value(matrix_all)
    prevalences = np.linspace(35, 55, 11)

    pred_er = pd.DataFrame(columns=['fpr', 'fnr'])

    for prev in prevalences:
        m = constant_predictive_value(ppv, npv, prev)
        pred_er.loc[prev] = error_rates(m)

    return pred_er


def plot_cpv_model(pred_er):
    """Plot error rates with constant predictive values.

    pred_er: DataFrame of predicted error rates
    """
    pred_er['fpr'].plot(label='Predicted FPR', color='C2')
    pred_er['fnr'].plot(label='Predicted FNR', color='C4')
    decorate(xlabel='Prevalence', ylabel='Percent',
             title='Error rates, constant predictive value')


def constant_error_rates(fpr, fnr, prev):
    """Make a confusion matrix with given metrics.

    fpr: false positive rate
    fnr: false negative rate
    prev: prevalence

    returns: confusion matrix
    """
    prev /= 100
    fpr /= 100
    fnr /= 100
    a = [[prev*(1 - fnr),  fpr*(1 - prev)],
         [fnr*prev,       (fpr - 1)*(prev - 1)]]

    return pd.DataFrame(a)


def run_cer_model(cp):
    """Run the constant error rate model.

    cp: DataFrame of COMPAS data

    returns: DataFrame with a row for each prevalence and
             a column for each metric (PPV, NPV)
    """
    matrix_all = make_matrix(cp)

    fpr, fnr = error_rates(matrix_all)
    prevalences = np.linspace(35, 65, 11)

    pred_pv = pd.DataFrame(columns=['ppv', 'npv'])

    for prev in prevalences:
        m = constant_error_rates(fpr, fnr, prev)
        pred_pv.loc[prev] = predictive_value(m)

    return pred_pv


def plot_cer_model(pred_pv):
    """Plot error rates with constant predictive values.

    pred_er: DataFrame of predicted error rates
    """
    pred_pv['ppv'].plot(label='Predicted PPV', color='C0')
    pred_pv['npv'].plot(label='Predicted NPV', color='C1')
    decorate(xlabel='Prevalence', ylabel='Percent',
             title='Predictive value, constant error rates')


from scipy.interpolate import interp1d

def interpolate(series, value, **options):
    """Evaluate a function at a value.

    series: Series
    value: number
    options: passed to interp1d (default is linear interp)

    returns: number
    """
    interp = interp1d(series.index, series.values, **options)
    return interp(value)


def crossing(series, value, **options):
    """Find where a function crosses a value.

    series: Series
    value: number
    options: passed to interp1d (default is linear interp)

    returns: number
    """
    interp = interp1d(series.values, series.index, **options)
    return interp(value)


def sweep_threshold(cp):
    """Sweep a range of threshold and compute accuracy metrics.

    cp: DataFrame of COMPAS data

    returns: DataFrame with one row for each threshold and
             one column for each metric
    """
    index = range(0, 11)
    columns = ['FPR', 'FNR', 'PPV', 'NPV', 'Prevalence']
    table = pd.DataFrame(index=index, columns=columns, dtype=float)

    for threshold in index:
        m = make_matrix(cp, threshold)
        metrics = compute_metrics(m)
        table.loc[threshold] = metrics['Percent']

    return table


def plot_roc(table, **options):
    """Plot the ROC curve.

    table: DataFrame of metrics as a function of
           classification threshold
    options: passed to plot
    """
    plt.plot([0,100], [0,100], ':', color='gray')
    sens = 100-table['FNR']
    plt.plot(table['FPR'], sens, **options)
    decorate(xlabel='FPR',
             ylabel='Sensitivity (1-FNR)',
             title='ROC curve')


from scipy.integrate import simps

def compute_auc(table):
    """Compute the area under the ROC curve.

    Uses the trapezoid rule, so
    """
    y = 100-table['FNR']
    x = table['FPR']
    y = y.sort_index(ascending=False) / 100
    x = x.sort_index(ascending=False) / 100
    return simps(y.values, x.values)
    
