from analyzer import Analyzer
import pandas as pd
import pytest 
from pandas import testing as pdt
import numpy as np
import mock

customer_payment_df = pd.DataFrame({ 
    'customer_id': [1, 1, 1, 2, 2],
    'payment_date': [pd.datetime(2020, 1, 1), pd.datetime(2020, 2, 1), pd.datetime(2020, 3, 1),  pd.datetime(2020, 1, 1), pd.datetime(2020, 4, 1)], 
    "amount" : [10] * 3 + [20] * 2,
    "cohort" : [pd.datetime(2020, 1, 1)] * 3 + [pd.datetime(2020, 1, 1)] * 2, 
    "payment_period" : [0, 1, 2, 0, 3], 
    "payment_frequency" : ["monthly"] * 3 + ["quartelry"] * 2,
})

spend_df = pd.DataFrame({
    "cohort" : [pd.datetime(2020, 1, 1)],
    "spend" : [100]
    })

az = Analyzer(customer_payment_df, spend_df)

def test_get_average_customer_acquisition_cost():
    result = az.get_average_customer_aquisition_cost()
    expected = spend_df["spend"].sum() / customer_payment_df["customer_id"].nunique()
    assert result == expected

def test_get_customer_count_by_payment_frequency():
    result = az.get_customer_count_by_payment_frequency()
    expected = pd.DataFrame({"monthly": [1], "quartelry": [1], "total" : [2]}, index = [pd.datetime(2020, 1, 1)])
    pdt.assert_frame_equal(result, expected, check_dtype=False, check_names=False)

def test_get_cohort_payments_by_payment_period():
    result = az.get_cohort_payments_by_payment_period()
    expected = pd.DataFrame({0: [30], 1: [40], 2: [50], 3: [70]}, index = [pd.datetime(2020, 1, 1)])
    pdt.assert_frame_equal(result, expected, check_dtype=False, check_names=False)

    result = az.get_cohort_payments_by_payment_period(should_cumulate=False)
    expected = pd.DataFrame({0: [30], 1: [10], 2: [10], 3: [20]}, index = [pd.datetime(2020, 1, 1)])
    pdt.assert_frame_equal(result, expected, check_dtype=False, check_names=False)

def test_get_cohort_size_by_payment_period():
    result = az.get_cohort_size_by_payment_period(should_cumulate=True)
    expected = pd.DataFrame({0: [2], 1: [3], 2: [4], 3: [5]}, index = [pd.datetime(2020, 1, 1)])
    pdt.assert_frame_equal(result, expected, check_dtype=False, check_names=False)  

    result = az.get_cohort_size_by_payment_period(should_cumulate=False)
    expected = pd.DataFrame({0: [2], 1: [1], 2: [1], 3: [1]}, index = [pd.datetime(2020, 1, 1)])
    pdt.assert_frame_equal(result, expected, check_dtype=False, check_names=False)  

def test_get_cohort_roas_by_payment_period():
    result = az.get_cohort_roas_by_payment_period(should_cumulate=False)
    expected = pd.DataFrame({0: [30/100, ], 1: [10/100], 2: [10/100], 3: [20/100]}, index = [pd.datetime(2020, 1, 1)])
    pdt.assert_frame_equal(result, expected, check_dtype=False, check_names=False)

    result = az.get_cohort_roas_by_payment_period(should_cumulate=True)
    expected = pd.DataFrame({0: [30/100], 1: [40/100], 2: [50/100], 3: [70/100]}, index = [pd.datetime(2020, 1, 1)])
    pdt.assert_frame_equal(result, expected, check_dtype=False, check_names=False)  

def test_get_cohort_payment_churns_by_payment_period():
    mock_payments_by_payments_period = pd.DataFrame({0: [100], 1: [50], 2: [25], 3: [50]}, index = [pd.datetime(2020, 1, 1)])
    with mock.patch.object(Analyzer, 'get_cohort_payments_by_payment_period', return_value=mock_payments_by_payments_period):
        result = az.get_cohort_payment_churns_by_payment_period()
        expected = pd.DataFrame({0: [np.NaN], 1: [0.5], 2: [0.5], 3: [-1]}, index = [pd.datetime(2020, 1, 1)])
        pdt.assert_frame_equal(result, expected, check_dtype=False, check_names=False)

############## TEST STATISTICS FUNCTIONS (NOT PLOT FUNCTIONS) ####################

def test_extrapolate_cohorts_by_payment_period_quarterly():
    cohorts = [pd.datetime(2020, 1, 1), pd.datetime(2020, 2, 1)]
    cohort_data_by_payment_period = pd.DataFrame(
        {0: [10, 30], 3 : [20, 40]}, 
        index = cohorts
    )
    churns = [0.1, 0.5, 0.25] # The first churn is for 1 -> 2, the second for 2 -> 3. Therefore this function will add 2 columns 2 and 3
    result = Analyzer.extrapolate_cohorts_by_payment_period(cohort_data_by_payment_period, churns, payment_period_distance = 3)
    expected_result = result.copy()
    expected_result[6] = [10, 20]
    expected_result[9] = [7.5, 15]

    pdt.assert_frame_equal(result, expected_result, check_dtype=False, check_names=False)

def test_extrapolate_cohorts_by_payment_period_monthly():
    cohorts = [pd.datetime(2020, 1, 1), pd.datetime(2020, 2, 1)]
    cohort_data_by_payment_period = pd.DataFrame(
        {0: [10, 30], 1 : [20, 40]}, 
        index = cohorts
    )
    churns = [0.1, 0.5, 0.25] # The first churn is for 1 -> 2, the second for 2 -> 3. Therefore this function will add 2 columns 2 and 3
    result = Analyzer.extrapolate_cohorts_by_payment_period(cohort_data_by_payment_period, churns, payment_period_distance = 1)
    expected_result = result.copy()
    expected_result[2] = [10, 20]
    expected_result[3] = [7.5, 15]

    pdt.assert_frame_equal(result, expected_result, check_dtype=False, check_names=False)

def test_extrapola_cohorts_by_payment_period_works_with_less_churns_than_columns_in_the_df():
    cohort_df = pd.DataFrame({0 : [20, 20, 20], 1 : [20, 20, np.NaN], 2 : [20, 20, np.NaN]})  
    result = Analyzer.extrapolate_cohorts_by_payment_period(cohort_df, [0.5], 1)
    expected_result = cohort_df.copy() 
    expected_result.iloc[2, 1] = 10
    assert result.columns.equals(expected_result.columns)
    pdt.assert_frame_equal(result, expected_result, check_dtype=False, check_names=False)
 
def test_extrapolate_single_cohort_with_not_enough_churn_values_to_predict():
    cohort = np.array([10, 20])
    churns = [0.5]
    result = Analyzer.extrapolate_single_cohort(cohort, churns)
    expected_result = [10, 20]
    assert np.array_equal(result, expected_result)

def test_extrapolate_single_cohort():
    cohort = np.array([10, 20])
    churns = [0.5, 0.5, 0.5] # The first churrn is for 1 -> 2, the second for 2 -> 3. Therefore this function will 2 entries to the cohort.
    result = Analyzer.extrapolate_single_cohort(cohort, churns) 
    expected_result = [10, 20, 10, 5]  
    assert expected_result

def test_add_cohort_dfs():
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df2 = pd.DataFrame({'A': [1, 1], 'B': [1, 1]})
    expected = pd.DataFrame({'A': [2, 3], 'B': [4, 5]})
    result = Analyzer.add_cohort_dfs(df1, df2)
    pdt.assert_frame_equal(result, expected)

def test_remove_entries_with_negative_payment_periods():
    customer_df = pd.DataFrame({'customer_id': [1, 2, 3], 'payment_period': [5, -2, 0]})
    expected = pd.DataFrame({'customer_id': [1, 3], 'payment_period': [5, 0]})
    result = Analyzer.remove_entries_with_negative_payment_periods(customer_df)
    pdt.assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))

def test_remove_all_nan_columns():
    df = pd.DataFrame({'A': [1, 2], 'B': [float('nan'), float('nan')], 'C': [3, 4]})
    expected = pd.DataFrame({'A': [1, 2], 'C': [3, 4]})
    result = Analyzer.remove_all_nan_columns(df)
    pdt.assert_frame_equal(result, expected)

def test_cumulate_df_across_columns():
    df = pd.DataFrame({'A': [1., 2.], 'B': [3., 4.], 'C': [5., 6.]})
    expected = pd.DataFrame({'A': [1., 2.], 'B': [4., 6.], 'C': [9., 12.]})
    result = Analyzer.cumulate_df_across_columns(df)
    pdt.assert_frame_equal(result, expected)

def test_uncumulate_df_across_columns():
    df = pd.DataFrame({'A': [1, 2], 'B': [4, 6], 'C': [9, 12]})
    expected = pd.DataFrame({'A': [1., 2.], 'B': [3., 4.], 'C': [5., 6.]})
    result = Analyzer.uncumulate_df_across_columns(df)
    pdt.assert_frame_equal(result, expected)

def test_center_df_by_col_mean():
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    expected = pd.DataFrame({'A': [-0.5, 0.5], 'B': [-0.5, 0.5]})
    result = Analyzer.center_df_by_col_mean(df)
    pdt.assert_frame_equal(result, expected)

def test_filter_wide_df_by_cohorts():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=pd.to_datetime(['2021-01-01', '2022-01-01', '2023-01-01']))
    min_cohort = pd.to_datetime('2022-01-01')
    max_cohort = pd.to_datetime('2023-01-01')
    expected = pd.DataFrame({'A': [2, 3], 'B': [5, 6]}, index=pd.to_datetime(['2022-01-01', '2023-01-01']))
    result = Analyzer.filter_wide_df_by_cohorts(df, min_cohort=min_cohort, max_cohort=max_cohort)
    pdt.assert_frame_equal(result, expected)

def test_divide_df_by_col():
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 6]})
    expected = pd.DataFrame({'A': [1.0, 1.0], 'B': [3.0, 3.0]})
    result = Analyzer.divide_df_by_col(df, 'A')
    pdt.assert_frame_equal(result, expected)

def test_get_cols_diff():
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
    expected = pd.DataFrame({'B': [2., 2.], 'C': [2., 2.]})
    result = Analyzer.get_cols_diff(df, drop_na_col=True)
    pdt.assert_frame_equal(result, expected)

def test_roll_wide_to_cal_wide():
    cohorts = [pd.datetime(2020, 1, 1), pd.datetime(2020, 2, 1)]
    wide_df = pd.DataFrame({0: [1, 2], 1: [4, 5]}, index=cohorts)
    
    result = Analyzer.roll_wide_to_cal_wide(wide_df, row_fill=False)
    expected_result = pd.DataFrame({
        pd.datetime(2020, 1, 1): [1., np.NaN],
        pd.datetime(2020, 2, 1): [4., 2.],
        pd.datetime(2020, 3, 1): [np.NaN, 5.]
    }, index=cohorts, columns=[pd.datetime(2020, 1, 1), pd.datetime(2020, 2, 1), pd.datetime(2020, 3, 1)])
    expected_result.columns.name = "datetime"
    print(result.columns)
    pdt.assert_frame_equal(result, expected_result, check_names=False)

def test_tall_df_to_wide_df():
    cohorts = [pd.datetime(2020, 1, 1), pd.datetime(2020, 2, 1)]
    wide_df = pd.DataFrame(
        {0 : [1, 2],  1 : [4, 5]}
    , index = cohorts)

    result = Analyzer.wide_rolling_to_tall_calendar(wide_df)
    expected_result = pd.DataFrame({
        'cohort': [pd.datetime(2020, 1, 1), pd.datetime(2020, 1, 1), pd.datetime(2020, 2, 1), pd.datetime(2020, 2, 1)],
        'datetime': [pd.datetime(2020, 1, 1), pd.datetime(2020, 2, 1), pd.datetime(2020, 2, 1), pd.datetime(2020, 3, 1)],
        'value': [1, 4, 2, 5]
    })
    
    pdt.assert_frame_equal(result, expected_result)

def test_wide_rolling_to_tall_calendar():
    cohorts = [pd.datetime(2020, 1, 1), pd.datetime(2020, 2, 1)]
    wide_df = pd.DataFrame({0: [1, 2], 1: [4, 5]}, index=cohorts)

    result = Analyzer.wide_rolling_to_tall_calendar(wide_df)
    expected_result = pd.DataFrame({
        'cohort': [pd.datetime(2020, 1, 1), pd.datetime(2020, 1, 1),
                   pd.datetime(2020, 2, 1), pd.datetime(2020, 2, 1)],
        'datetime': [pd.datetime(2020, 1, 1), pd.Timestamp(2020, 2, 1),
                     pd.datetime(2020, 2, 1), pd.datetime(2020, 3, 1)],
        'value': [1, 4, 2, 5]
    })
    
    pdt.assert_frame_equal(result, expected_result)