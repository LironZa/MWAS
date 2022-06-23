from collections import namedtuple
import numpy as np
import statsmodels.api as sm
from pandas import DataFrame


def empty_res(res, x, ret_cov_fields):
    res.R2 = 0
    res.Coef = 0
    res.Pval = 1
    res.Coef_025 = 0
    res.Coef_975 = 0
    if ret_cov_fields:
        for cov in x.columns[:-1]:
            setattr(res, cov + '_Pval', 7)
            setattr(res, cov + '_Coef', 7)
        setattr(res, 'MAF_Pval', 7)
        setattr(res, 'MAF_Coef', 7)

    res.Coef_SE = np.NaN
    res.MAF_SE = np.NaN
    res.maj_n = np.NaN
    res.maj_mn = np.NaN
    res.maj_md = np.NaN
    res.maj_sd = np.NaN
    res.min_n = np.NaN
    res.min_mn = np.NaN
    res.min_md = np.NaN
    res.min_sd = np.NaN
    return res


def ols(xy, xy_col_inds, test_x_col_ind, add_constant=True, logistic_regression=False, is_y_valid_f=None,
        verbosity=0, collect_data=False, illegal_test_x_val=None, ret_cov_fields=False):
    assert isinstance(xy, DataFrame)

    if illegal_test_x_val is not None:
        local_xy = xy.iloc[:, xy_col_inds][xy.iloc[:, xy_col_inds[test_x_col_ind]] != illegal_test_x_val]
        x = local_xy.iloc[:, :-1]
        y = local_xy.iloc[:, -1]
    else:
        x = xy.iloc[:, xy_col_inds[:-1]]
        y = xy.iloc[:, xy_col_inds[-1]]

    if is_y_valid_f is not None and not is_y_valid_f(y):
        return None

    if add_constant:
        x = sm.add_constant(x)

    if collect_data:
        return x.join(y)

    res_fields = ['N', 'R2', 'Coef', 'Pval', 'Coef_025', 'Coef_975',
                  'Coef_SE', 'MAF_SE', 'maj_n', 'maj_m', 'maj_sd', 'min_n', 'min_m', 'min_sd']

    if ret_cov_fields:
        for cov in x.columns[:-1]:
            res_fields += ['{}_Pval'.format(cov), '{}_Coef'.format(cov)]
        res_fields += ['MAF_Pval', 'MAF_Coef']

    res = namedtuple('OLS', res_fields)

    res.N = len(x)

    try:
        if logistic_regression:
            o = sm.Logit(y, x).fit(disp=verbosity)
            if not o.mle_retvals['converged']:
                o = sm.Logit(y, x).fit(disp=verbosity, method='nm')
                if not o.mle_retvals['converged']:
                    return None
                    # return empty_res(res, x, ret_cov_fields)
        else:
            o = sm.OLS(y, x).fit(disp=verbosity)

        res.R2 = o.prsquared if logistic_regression else o.rsquared
        res.Coef = o.params.iloc[-1]  # iloc[-1] == ['MAF']
        res.Pval = o.pvalues.iloc[-1]
        res.Coef_SE = o.bse.iloc[-1]
        res.MAF_SE = x.iloc[:, -1].std()
        conf_int = o.conf_int()
        res.Coef_025 = conf_int.iloc[-1][0]
        res.Coef_975 = conf_int.iloc[-1][1]
        if ret_cov_fields:
            for cov in x.columns[:-1]:  # additional data, for the covariates
                setattr(res, cov + '_Pval', o.pvalues[cov])
                setattr(res, cov + '_Coef', o.params[cov])
            setattr(res, 'MAF_Pval', o.pvalues.iloc[-1])
            setattr(res, 'MAF_Coef', o.params.iloc[-1])


        MAF_1_VALUE = np.uint8(200)
        maj_x, maj_y = x.loc[x.iloc[:, -1] > .5*MAF_1_VALUE], y.loc[x.iloc[:, -1] > .5*MAF_1_VALUE]
        res.maj_n = len(maj_y)
        res.maj_mn = maj_y.mean()
        res.maj_md = maj_y.median()
        res.maj_sd = maj_y.std()

        min_x, min_y = x.loc[x.iloc[:, -1] <= .5*MAF_1_VALUE], y.loc[x.iloc[:, -1] <= .5*MAF_1_VALUE]
        res.min_n = len(min_y)
        res.min_mn = min_y.mean()
        res.min_md = min_y.median()
        res.min_sd = min_y.std()
        assert len(min_y) + len(maj_y) == len(y)

    except BaseException as e:
        return None
        # res = empty_res(res, x, ret_cov_fields)

    return res
