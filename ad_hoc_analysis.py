from analyzer import Analyzer
import pandas as pd

############################### USED IN AD COH ANALYSIS + BACKTEST #########################################################

def calculate_churn_stats(churn_data, excluded_cohort, cohort_periods, projection_length, min_number_of_points_stabilization_period=11, percentile_values=[.5, .75, .9]):
    """
    Calculate churn statistics for multiple cohorts and periods with flexible percentile values
    and projection length, dynamically handling specific periods (like C1, C2, C3) and extending the last period if necessary.

    Parameters:
    - churn_data: DataFrame containing churn data (monthly or quarterly).
    - excluded_cohort: List of cohorts (timestamps) to exclude.
    - cohort_periods: Dictionary where keys are cohort labels (e.g., 'C1', 'C2', 'C3'), and values are tuples defining date ranges.
    - projection_length: Integer indicating how many months or quarters to project for churn.
    - min_number_of_points_stabilization_period: Minimum number of data points required to include a period in the stabilization calculation (default is 11).
    - percentile_values: List of percentiles to compute (default: [.5, .75, .9]).

    Returns:
    - median_churns: List of median churns for each period.
    - conservative_churns: List of 75th percentile churns for each period.
    - very_conservative_churns: List of 90th percentile churns for each period.
    """
    
    # Convert percentile values to their string counterparts ('50%', '75%', '90%', etc.)
    percentile_labels = [f'{int(p * 100)}%' for p in percentile_values]

    median_churns = []
    conservative_churns = []
    very_conservative_churns = []

    # Process specific periods (e.g., C1, C2, C3 for monthly or C3, C6 for quarterly)
    stabilization_label = None  # To track stabilization period (e.g., 'C4+')
    for index, (label, period) in enumerate(cohort_periods.items()):
        # Check if the period is a stabilization period (ends with '+')
        if '+' in label:
            stabilization_label = label
            stabilization_period = period  # This will be used for the stabilization calculation
            break
        
        # Extract min and max cohort dates
        min_cohort, max_cohort = period
        relevant_churns = Analyzer.filter_wide_df_by_cohorts(
            churn_data, min_cohort=min_cohort, max_cohort=max_cohort, cohorts_to_exclude=excluded_cohort
        )

        # Check if describe returns a DataFrame or Series
        stats = relevant_churns.describe(percentiles=percentile_values)
        # If we have a DataFrame, process the appropriate column
        stats = stats.iloc[:, index]  # Dynamically select the correct column based on the index
        # Append calculated values to respective lists
        median_churns.append(stats[percentile_labels[0]])
        conservative_churns.append(stats[percentile_labels[1]])
        very_conservative_churns.append(stats[percentile_labels[2]])

    # Handle the stabilization period if present
    if stabilization_label:
        # Get min_cohort and max_cohort from the stabilization period
        min_cohort, max_cohort = stabilization_period

        # Reuse the last period's DataFrame for stabilization
        relevant_churns = Analyzer.filter_wide_df_by_cohorts(
                churn_data, min_cohort=min_cohort, max_cohort=max_cohort, cohorts_to_exclude=excluded_cohort
        )

        # Get describe statistics for stabilization period
        stabilization_stats = relevant_churns.describe(percentiles=percentile_values)
        
        # Filter out columns where the count is less than the minimum number of points
        counts = stabilization_stats.loc['count']
        valid_columns = counts[counts >= min_number_of_points_stabilization_period].index
        stabilization_stats_filtered = stabilization_stats[valid_columns]

        # Drop columns with NaN values
        stabilization_stats_filtered = stabilization_stats_filtered.dropna(axis=1)

        # Filter out columns before stabilization period
        stabilization_stats_filtered = stabilization_stats_filtered.iloc[:, len(cohort_periods)-1:]
        
        # Compute the average across the remaining columns
        stabilization_avg = stabilization_stats_filtered.mean(axis=1, skipna=True)
        
        # Append the same average value for the remaining projection periods
        num_projection_periods = projection_length - len(cohort_periods) + 1  # +1 to account for stabilization period
        median_churns += [stabilization_avg[percentile_labels[0]]] * num_projection_periods
        conservative_churns += [stabilization_avg[percentile_labels[1]]] * num_projection_periods
        very_conservative_churns += [stabilization_avg[percentile_labels[2]]] * num_projection_periods

    return median_churns, conservative_churns, very_conservative_churns