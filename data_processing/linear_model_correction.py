import numpy as np
import os
import pickle
import statsmodels.formula.api as smf


ALL_TARGETS = ['AllMeters', 'AllMeters_Rwc', 'Food', 'KCal_hr',
               'PedMeters', 'PedMeters_Rwc', 'PedSpeed', 'RQ', 'VCO2', 'VH2O',
               'VO2', 'Water', 'WheelMeters', 'WheelSpeed', 'XBreak', 'YBreak',
               'ZBreak']
GAS_TARGETS = ['RQ', 'KCal_hr', 'VCO2', 'VH2O', 'VO2']


def control_covariate(df, covariate, targets, interaction_terms,
                      reference=None):
    df = df.copy()
    if reference is None:
        reference = df[covariate].mean()

    # Subsample to speed up fitting the BW correction
    np.random.seed(42)
    mask = np.random.randn(len(df)) > 1.0
    subsample_df = df[mask].copy()

    correction_models = {}
    interaction_vars = [int_term[0] for int_term in interaction_terms]
    for target in targets:
        # Fit correction model
        reg_df = subsample_df[[target, covariate] + interaction_vars]
        formula_terms = [f"{target} ~ {covariate}"]

        for interaction_term in interaction_terms:
            interaction_var, reference_level = interaction_term
            interaction_formula = (
                f"C({interaction_var},Treatment(reference={reference_level}))")
            formula_terms.append(f"{covariate}*{interaction_formula}")

        formula = " + ".join(formula_terms)
        model = smf.ols(formula, data=reg_df).fit()

        # Correct values to reference
        baseline_df = df[[target, covariate] + interaction_vars]
        standardized_df = df[[target, covariate] + interaction_vars].copy()
        standardized_df.loc[:, covariate] = reference
        correction = (
            model.predict(baseline_df) - model.predict(standardized_df)).values
        df.loc[:, target] = df[target] - correction
        correction_models[target] = model
    return df, correction_models


def control_for_run_number(df):
    return control_covariate(
        df, 'run_number', ALL_TARGETS, [('states', 0)],
        reference='run_01')


def control_for_body_mass(df):
    return control_covariate(
      df, 'BodyMass', GAS_TARGETS, [('states', 0), ('age_in_months', 6)])
