import jax.numpy as jnp
from jax import grad
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from typing import Callable, Optional, Tuple
import xgboost as xgb

from .distributions import make_hierarchical_logbeta_prior_log_pdf


def caspar_loss(
    ypred: jnp.ndarray,
    age: jnp.ndarray,
    survival_time: jnp.ndarray,
    pdf_Z: Callable,
    log_pdf_logbeta: Callable,
    animal_ids: Optional[jnp.ndarray] = None,
    truncation_time: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:

    # Group observations from the same mouse if mouse IDs are available
    if animal_ids is not None:
        codes, _ = pd.factorize(animal_ids)
        num_labels = jnp.max(codes) + 1
        animal_indicators = jnp.eye(num_labels)[codes]
    else:
        # Assume that each obs is from a distinct mouse
        animal_indicators = jnp.eye(len(ypred))

    # Get survival status
    censored = survival_time < 0
    survival_time = jnp.abs(survival_time)

    assert jnp.all(jnp.abs(survival_time) >= age)
    if truncation_time is not None:
        assert jnp.all(age >= truncation_time)

    estimated_log_beta = jnp.log(age) - ypred
    log_beta_prob = log_pdf_logbeta(estimated_log_beta, animal_indicators)
    log_survival_time = jnp.log(survival_time)
    s_y = log_survival_time - estimated_log_beta

    if truncation_time is not None:
        log_truncation_time = jnp.log(truncation_time)
        t_y = log_truncation_time - estimated_log_beta
        uncensored_likelihood = pdf_Z(s_y[~censored], t_y[~censored])
    else:
        uncensored_likelihood = pdf_Z(
                s_y[~censored], jnp.zeros_like(s_y[~censored]))

    loglikelihood = 0
    loglikelihood += jnp.sum(jnp.log(uncensored_likelihood + 1e-12))
    loglikelihood += log_beta_prob
    return -loglikelihood


def make_objective_and_metric_function(
    label_df: pd.DataFrame,
    pdf_Z: Callable,
    log_pdf_logbeta: Callable,
    age_var: str = "age",
    survival_var: str = "survival",
    animal_id_var: Optional[str] = None,
    truncation_var: Optional[str] = None,
) -> tuple[Callable, Callable]:
    def _get_labels(label_indices: jnp.ndarray):
        age = label_df.loc[label_indices, age_var].values
        survival_time = label_df.loc[label_indices, survival_var].values

        if truncation_var is not None:
            truncation_time = (label_df.loc[label_indices, truncation_var]
                                       .values)
        else:
            truncation_time = None

        if animal_id_var is not None:
            animal_ids = label_df.loc[label_indices, animal_id_var].values
        else:
            animal_ids = None

        return age, survival_time, truncation_time, animal_ids

    def _loss_fn(ypred: jnp.ndarray, data: xgb.DMatrix) -> jnp.ndarray:
        label_indices = data.get_label()
        labels = _get_labels(label_indices)
        age, survival_time, truncation_time, animal_ids = labels
        return caspar_loss(
            ypred,
            age,
            survival_time,
            pdf_Z,
            log_pdf_logbeta,
            truncation_time=truncation_time,
            animal_ids=animal_ids,)

    def gradient(x, y):
        g = jnp.clip(grad(_loss_fn)(x, y), -15, 15)
        return g

    # XGBoost takes a diagonal approximation to the hessian
    hessian = grad(lambda x, y: jnp.sum(gradient(x, y)))

    def caspar_xgb_obj(
        ypred: jnp.ndarray, data: xgb.DMatrix
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return gradient(ypred, data), hessian(ypred, data)

    def caspar_xgb_metric(ypred: jnp.ndarray, data: xgb.DMatrix):
        label_indices = data.get_label()
        labels = _get_labels(label_indices)
        age, survival_time, truncation_time, animal_ids = labels

        r, _ = pearsonr(np.exp(ypred), age)
        neg_loglikelihood = _loss_fn(ypred, data) / len(ypred)
        censored = survival_time < 0
        time_to_death = jnp.abs(survival_time) - age
        r_chron, _ = pearsonr(age[~censored], time_to_death[~censored])
        r_bio, _ = pearsonr(np.exp(ypred)[~censored], time_to_death[~censored])
        return [("neg-ll", neg_loglikelihood),
                ("age_corr", r),
                ("ttd_corr", -r_bio)]
    return caspar_xgb_obj, caspar_xgb_metric


def xgb_train(
    dtrain: xgb.DMatrix,
    dtest: xgb.DMatrix,
    label_df: pd.DataFrame,
    pdf_Z: Callable,
    beta_sigma: float,
    eps_sigma: float,
    age_var: str = "age",
    survival_var: str = "survival",
    animal_id_var: Optional[str] = None,
    truncation_var: Optional[str] = None,
    verbose_eval=100,
    num_boost_round=5000,
    **xgb_params
):
    all_xgb_params = {
        "disable_default_eval_metric": 1,
        "min_split_loss": 0,
        "min_child_weight": 5,
        "base_score": 5,
        "learning_rate": 5e-3,
        "subsample": 0.8,
        "colsample_bynode": 0.8,
        "lambda": 1e-5,
        "max_depth": 4,
    }
    all_xgb_params.update(xgb_params)
    log_pdf_logbeta = make_hierarchical_logbeta_prior_log_pdf(
            0, beta_sigma, eps_sigma)

    obj_fn, metric_fn = make_objective_and_metric_function(
        label_df, pdf_Z, log_pdf_logbeta,
        age_var=age_var, survival_var=survival_var,
        animal_id_var=animal_id_var, truncation_var=truncation_var)

    current_eval_results = {}
    caspar_model = xgb.train(
        all_xgb_params,
        dtrain,
        obj=obj_fn,
        feval=metric_fn,
        num_boost_round=num_boost_round,
        evals_result=current_eval_results,
        evals=[(dtrain, "train"), (dtest, "test")],
        verbose_eval=verbose_eval,
    )
    return caspar_model, current_eval_results
