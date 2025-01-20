import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import itertools
import warnings
import numpy as np
from typing import Union

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set global figure size (width, height) in inches
plt.rcParams['figure.figsize'] = (16, 14)

class Analyzer:
    """
    The Analyzer object that you'll use to do structured analysis on a company's data.
    Most of the heavy lifting should be done in this class / in the code so that the notebook is more easily interpretable.
    Create whatever functions you think are useful in order to analyze the data.
    Whatever you create here should be useful when we get another company's data.
    To be clear--the existing code is an example (and not a great one). Feel free to remove existing code / functions.
    Make sure to keep code legible / understandable!
    """

    def __init__(self, customer_payment_df, spend_df) -> None:
        """
        Initializes the Analyzer--feel free to do whatever preprocessing you feel appropriate.
        """
        for col in ["cohort", "spend"]:
            assert col in spend_df.columns
        for col in ["customer_id", "payment_date", "amount", "cohort", "payment_period"]:
            assert col in customer_payment_df.columns

        self.spend_df = spend_df.set_index("cohort")
        self.first_cohort_with_spend = self.spend_df.index.min()
        self.customer_payment_df =  self.remove_entries_with_negative_payment_periods(customer_payment_df)
    
    def fill_in_missing_spend(self) -> None:
        """
        Fills in missing spend data by estimating the spend based on the average customer acquisition cost.
        """
        initial_cohort_sizes = self.get_cohort_size_by_payment_period(should_cumulate=False)[0]
        cohorts_sizes_without_spend = initial_cohort_sizes[~initial_cohort_sizes.index.isin(self.spend_df.index)]
        estimated_spend_df = pd.DataFrame(cohorts_sizes_without_spend * self.get_average_customer_aquisition_cost())
        estimated_spend_df.columns = ["spend"]
        self.spend_df = pd.concat([self.spend_df, estimated_spend_df], axis = 0)
        self.spend_df = self.spend_df.sort_index()

    def get_average_customer_aquisition_cost(self) -> float:
        """
        Returns the average customer acquisition cost.
        """
        total_spend = self.spend_df["spend"].sum()
        total_customers = self.customer_payment_df[self.customer_payment_df["cohort"] >= self.spend_df.index.min()]["customer_id"].nunique()
        return total_spend / total_customers
    
    def get_customer_count_by_payment_frequency(self) -> pd.DataFrame:
        """
        Groups by cohort and payment frequency and counts the number of unique customer ids.
        Returns a dataframe with cohort as index, a column for each payment frequency, and an additional column that sums up all the payment frequency columns.
        """
        grouped_df = self.customer_payment_df.groupby(["cohort", "payment_frequency"])["customer_id"].nunique()
        pivot_df = grouped_df.unstack(fill_value=0)
        pivot_df["total"] = pivot_df.sum(axis=1)
        return pivot_df
        
    def get_cohort_payments_by_payment_period(self, should_cumulate=True) -> pd.DataFrame:
        """
        Returns a dataframe containing payments per cohort every payment period.
        should_cumulate: bool, default=True, whether to cumulate the payments.
        """
        grouped_df = self.customer_payment_df.groupby(["cohort", "payment_period"])["amount"].sum()
        grouped_df = grouped_df.reset_index().pivot(index="cohort", columns="payment_period", values="amount")
        if should_cumulate:
            return Analyzer.cumulate_df_across_columns(grouped_df)
        return grouped_df

    def get_cohort_size_by_payment_period(self, should_cumulate=True) -> pd.DataFrame:
        """
        Returns a dataframe containing the number of customers per cohort every payment period.
        should_cumulate: bool, default=True, whether to cumulate the payments.
        """
        grouped_df = self.customer_payment_df.groupby(["cohort", "payment_period"])["customer_id"].nunique()
        grouped_df = grouped_df.reset_index().pivot(index="cohort", columns="payment_period", values="customer_id")
        if should_cumulate:
            return Analyzer.cumulate_df_across_columns(grouped_df)
        return grouped_df
    
    def get_number_of_churned_customers_by_payment_period(self, drop_na_cols=False) -> pd.DataFrame:
        """
        Returns a dataframe containing the number of churned customers per cohort every payment period. The number of customers that left at any pyament period.
        """
        cohort_sizes = self.get_cohort_size_by_payment_period(should_cumulate=False)
        # change sign so that drops are positive
        return  -1 * Analyzer.get_cols_diff(cohort_sizes, drop_na_col=drop_na_cols)
    
    def get_cohort_roas_by_payment_period(self, should_cumulate=True) -> pd.DataFrame:   
        """
        Returns a dataframe containing ROAS per cohort every payment period.
        should_cumulate: bool, default=True, whether to cumulate the payments.
        """
        cohort_payments = self.get_cohort_payments_by_payment_period(should_cumulate=should_cumulate)
        # We can obly calculate ROAS for cohorts that have spend data
        cohort_payments = Analyzer.filter_wide_df_by_cohorts(cohort_payments, min_cohort=min(self.spend_df.index))
        cohort_spend = self.spend_df["spend"]
        roas = cohort_payments.div(cohort_spend, axis=0)
        return Analyzer.remove_all_nan_columns(roas)

    def get_m_n_values(self, N) -> pd.Series:   
        """
        Returns the M[N] per cohort.
        """
        return self.get_cohort_roas_by_payment_period(should_cumulate=True)[N]

    def get_delta_m_n_values(self, N) -> pd.Series:
        """
        Returns the delta M[N] per cohort.
        """
        return self.get_cohort_roas_by_payment_period(should_cumulate=False)[N]
    
    def get_cohort_payment_churns_by_payment_period(self) -> pd.DataFrame:
        """
        Returns a dataframe containing payment churns per cohort every payment period.
        """
        uncumulated_payments = self.get_cohort_payments_by_payment_period(should_cumulate=False)
        churns = -1 * uncumulated_payments.pct_change(axis=1, fill_method=None) #change sign so that percentage drops are positive
        return churns
    
    def get_cohort_size_churns_by_payment_period(self) -> pd.DataFrame:
        """
        Returns a dataframe containing size churns per cohort every payment period. Size as in number of customers.
        """
        uncumulated_sizes = self.get_cohort_size_by_payment_period(should_cumulate=False)
        size_churns = -1 * uncumulated_sizes.pct_change(axis=1, fill_method=None)    
        return size_churns
    
    ############################# EXTRAPOLATION FUNCTIONS #############################

    @staticmethod
    def extrapolate_cohorts_by_payment_period(cohort_data_by_payment_period_df, churns, payment_period_distance) -> pd.DataFrame:
        """
        Extrapolates a wide DataFrame of cohorts by payment period using the provided churns list.   
        the churns list should have all the churn values starting from period 1 up to the point that the user wants to project the cohort. 
        The cohort data should start from payment period 0 and have successive payment periods that align with the payment period distance (No gaps). 
        """
        # Result has one more column than the length of the churns list because there is no churn for the 0 payment period
        # If there are less churns provided than columns in the df, then the resulting df will have the same number of columns as the df.
        results_length = max(len(churns) + 1, len(cohort_data_by_payment_period_df.columns))
        extrapolated = pd.DataFrame(index=cohort_data_by_payment_period_df.index, columns=range(results_length))    
        for index, row in cohort_data_by_payment_period_df.iterrows():
            exisiting_and_extrapolated = Analyzer.extrapolate_single_cohort(row.values, churns)
            # If the extrapolated values are less than the required length, pad with NaNs - This can happen if the churns provided are not enough to project some cohorts that aleady have data up to that pont.
            if len(exisiting_and_extrapolated) < results_length:    
                exisiting_and_extrapolated = np.pad(exisiting_and_extrapolated, (0, results_length - len(exisiting_and_extrapolated)), mode='constant', constant_values=np.nan)
            extrapolated.loc[index] = exisiting_and_extrapolated

        extrapolated.columns = list(range(min(cohort_data_by_payment_period_df.columns), payment_period_distance * results_length + min(cohort_data_by_payment_period_df.columns), payment_period_distance))
        return extrapolated
    
    @staticmethod
    def extrapolate_single_cohort(cohort:Union[np.ndarray, pd.Series], churns:list) -> np.array:
        """
        Extrapolates a single cohort using provided churns list.
        the churns list should have all the churn values starting from period 1 up to the point that the user wants to project the cohort. 
        Churn values that correspond to periods that cohort has data for will not be used.
        The funtion locates the first value that needs to be projected and select the corresponding churn to apply.  
 
        returns a numpy array with the existing cohort values and the extrapolated values.
        """
        # locate the first non-nan index in the cohort if it exists else set to 0
        first_na_index = np.where(np.isnan(cohort))[0][0] if np.isnan(cohort).any() else 0
        # Locate the last non-nan value in the cohort and the corresponding churn value to apply for a one step projection.
        if first_na_index != 0:
            last_non_na_value = cohort[first_na_index - 1]
            churns = churns[(first_na_index-1):] 
        else:
            last_non_na_value = cohort[-1] 
            churns = churns[(len(cohort)-1):]
        
        # project the cohort data
        percentage_retained = 1 - np.array(churns)
        cumprod_percentage_retained = np.cumprod(percentage_retained)
        extrapolated_values = last_non_na_value * cumprod_percentage_retained
        # combine the existing and extrapolated values
        existing_and_extrapolated = np.concatenate([cohort[~np.isnan(cohort)], extrapolated_values])
        return existing_and_extrapolated
    
    ############################# GENERAL USE FUNCTIONS #############################
    
    @staticmethod
    def add_cohort_dfs(df1, df2, fill_value=0) -> pd.DataFrame:
        """
        Adds two DataFrames together.
        fill_value: int, default=0, the value to fill missing values with.
        """
        return df1.add(df2, fill_value=fill_value)
    
    @staticmethod
    def remove_entries_with_negative_payment_periods(customer_payment_df) -> pd.DataFrame:
        """
        Removes entries with negative payment periods.
        """
        return customer_payment_df[customer_payment_df["payment_period"] >= 0]
    
    @staticmethod
    def remove_all_nan_columns(df) -> pd.DataFrame:
        """
        Removes columns that have all entries as NaN.
        """
        return df.dropna(axis=1, how='all')
    
    @staticmethod
    def cumulate_df_across_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a dataframe containing the cumulative sum of the input dataframe.
        """
        return df.cumsum(axis=1)
    
    @staticmethod
    def uncumulate_df_across_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a dataframe containing the uncumulated values of the input dataframe.  
        """
        return df.diff(axis=1).fillna(df.iloc[:, 0:1])
    
    @staticmethod
    def center_df_by_col_mean(df: pd.DataFrame) -> pd.DataFrame:
        """
        Centers the DataFrame by the column means.
        """
        return  df - df.mean()

    @staticmethod
    def filter_wide_df_by_cohorts(wide_df, min_cohort=None, max_cohort=None, cohorts_to_exclude=[]) -> pd.DataFrame:
        """
        Filters a wide DataFrame by cohorts.
        min_cohort: pd.datetime, optional, minimum cohort to include based on index.
        max_cohort: pd.datetime, optional, maximum cohort to include based on index.
        cohorts_to_exclude: list[pd.datetime], optional, list of cohorts to exclude.
        """
        if min_cohort is not None:
            wide_df = wide_df[wide_df.index >= min_cohort]
        if max_cohort is not None:
            wide_df = wide_df[wide_df.index <= max_cohort]
        if len(cohorts_to_exclude) > 0:
            wide_df = wide_df[~wide_df.index.isin(cohorts_to_exclude)]
        return wide_df  

    @staticmethod
    def divide_df_by_col(df, col_name) -> pd.DataFrame:
        """
        Divides all columns by the given column.
        """
        return df.div(df[col_name], axis=0)
    
    @staticmethod
    def get_cols_diff(df, drop_na_col) -> pd.DataFrame:
        """
        Returns the difference between columns in a DataFrame.
        drop_na_col: bool, whether to drop the first column.    
        """
        diff = df.diff(axis=1)
        if drop_na_col:
            diff = diff.iloc[:, 1:]
        return diff
    
    @staticmethod   
    def roll_wide_to_cal_wide(roll_wide_df, row_fill=True) -> pd.DataFrame: 
        """
        Converts a wide rolling DataFrame to a wide calendar DataFrame.

        Term explanations:
        wide df: DataFrame with cohorts as index. 
        roll_wide : DataFrame with cohorts as index and payment periods as columns.  (roll for rolling months)
        cal_wide : DataFrame with cohorts as index and datetime as columns. (cal for calendar months)

        row_fill (bool): Determines whether to apply forward filling of NaN values row-wise. 
            If True, each NaN value in a row will be replaced by the last non-NaN value encountered in that row, starting from left to right. 
            If False, no filling will be applied, and NaN values will remain unchanged. This is useful for quarterly data as a lot of calendar months will be NaN.
        """
        tall_cal = Analyzer.wide_rolling_to_tall_calendar(roll_wide_df, value_name = "v")
        cal_wide = Analyzer.tall_df_to_wide_df(tall_cal, value_name = "v", row_fill=row_fill)    
        return cal_wide 
    
    @staticmethod
    def wide_rolling_to_tall_calendar(wide_roll, value_name="value") -> pd.DataFrame:
        """
        Converts a wide rolling DataFrame to a tall calendar DataFrame. 
        tall df: DataFrame with columns cohort, datetime (calendar month), value_name (value_name is the description for the values in the wide df)
        
        for example, the entry in the (wide roll) payments_by_payment_period DataFrame: payments_by_payment_period[2024-01-01, 2] = 100 
        would become it's own row in the tall calendar DataFrame with cohort = 2024-01-01, datetime = 2024-03-01, value_name = 100
        """
        rows_list = []
        for cohort in wide_roll.index:
            values = wide_roll.loc[cohort]
            months= list(wide_roll.columns)
            datetimes = [cohort + pd.offsets.MonthBegin(month) for month in months] 
            rows_list.append(pd.DataFrame({"cohort": [cohort] * len(values), "datetime": datetimes, value_name: values}))

        tall_calendar =  pd.concat(rows_list, ignore_index=True)
        tall_calendar = tall_calendar.dropna(subset=[value_name])
        return tall_calendar

    @staticmethod   
    def tall_df_to_wide_df(tall_df, value_name = "v", row_fill=False) -> pd.DataFrame:
        """
        Converts a tall DataFrame to a wide DataFrame.

        row_fill (bool): Determines whether to apply forward filling of NaN values row-wise. 
            If True, each NaN value in a row will be replaced by the last non-NaN value encountered in that row, starting from left to right. 
            If False, no filling will be applied, and NaN values will remain unchanged. This is useful for quarterly data as a lot of calendar months will be NaN.
        """
        wide_df = tall_df.pivot_table(index='cohort', columns='datetime', values=value_name)
        if row_fill:
            wide_df = wide_df.apply(lambda row: row.ffill(), axis=1)
        return wide_df
    
    ############################# PLOTTING FUNCTIONS #############################

    def plot_customer_count_and_spend_by_cohort(self, min_cohort=None,  title='Customer Count and Spend by Cohort', xlabel='Cohort', ylabel='Customer Count (Thousands)', fontsize=12) -> None:
        """
        Plots the customer count by payment frequency for each cohort, with spend data on a secondary axis.
        
        Parameters:
        - min_cohort (datetime, optional): The minimum cohort date to include in the plot.
        - title (str, optional): The title of the plot.
        - xlabel (str, optional): The label for the x-axis.
        - ylabel (str, optional): The label for the y-axis.
        - fontsize (int, optional): The font size for text in the plot.
        """
        # Retrieve customer count by payment frequency
        customer_count_df = self.get_customer_count_by_payment_frequency()
        
        # Rename customer count columns for clarity
        customer_count_df = customer_count_df.rename(columns={
            'total': 'Total Customers',
            'monthly': 'Monthly Customers',
            'quarterly': 'Quarterly Customers'
        })
        
        # Align indices of both DataFrames
        aligned_spend_df = self.spend_df.reindex(customer_count_df.index)
        # Rename spend columns for clarity
        aligned_spend_df = aligned_spend_df.rename(columns={
            'performance_marketing': 'Performance Marketing Spend',
            'brand_marketing': 'Brand Marketing Spend',
            'spend': 'Total Spend'
        })

        # Filter by min_cohort if specified
        if min_cohort is not None:
            customer_count_df = customer_count_df[customer_count_df.index >= min_cohort]
            aligned_spend_df = aligned_spend_df[aligned_spend_df.index >= min_cohort]

        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot customer count as a bar chart
        customer_count_df.plot(kind='bar', stacked=False, ax=ax)
        ax.set_title(title, fontsize=fontsize + 2)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)

        # Format x-axis as Month-Year
        ax.set_xticklabels(customer_count_df.index.strftime("%Y-%m"), rotation=45, ha='right', fontsize=fontsize - 2)

        # Scale customer count to thousands
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))

        # Set up secondary y-axis for spend data
        ax2 = ax.twinx()
        x_ticks = ax.get_xticks()

        # Define custom colors for spend lines (pink, purple, teal)
        spend_colors = ['#ff69b4', '#800080', '#008080']  # Pink, Purple, Teal

        # Plot each spend column on the secondary y-axis with custom colors
        for idx, column in enumerate(aligned_spend_df.columns):
            ax2.plot(
                x_ticks,
                aligned_spend_df[column],
                label=f'{column}',
                linestyle='--',
                marker='o',
                color=spend_colors[idx]
            )

        ax2.set_ylabel('Spend (Millions)', fontsize=fontsize)

        # Scale spend to millions
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

        # Place both legends side by side on the upper left
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            handles=handles1 + handles2,
            labels=labels1 + labels2,
            loc='upper left',
            fontsize=fontsize - 4
        )

        # Adjust tick parameters for both axes
        ax.tick_params(axis='both', which='major', labelsize=fontsize - 4)
        ax2.tick_params(axis='y', which='major', labelsize=fontsize - 4)

        # Adjust layout for better fit
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_wide_df(wide_df, min_cohort=None, add_legend=True, plot_cols=True, colormap="viridis", show=True,
                    title=None, x_label=None, y_label=None, font_size=14):
        """
        Plots a wide-format DataFrame with options to customize the title, labels, and formatting.

        Parameters:
        - wide_df (pd.DataFrame): The DataFrame to plot.
        - min_cohort (str, optional): Minimum cohort to include based on index.
        - add_legend (bool, optional): Whether to add a legend.
        - plot_cols (bool, optional): Whether to plot columns as lines.
        - colormap (str, optional): Colormap for the lines.
        - show (bool, optional): Whether to display the plot immediately.
        - title (str, optional): The title of the plot. If None, no title will be displayed.
        - x_label (str, optional): The label for the x-axis. If None, no label will be displayed.
        - y_label (str, optional): The label for the y-axis. If None, no label will be displayed.
        - font_size (int, optional): Font size for title, axis labels, and ticks.
        """
        if min_cohort is not None:
            wide_df = wide_df[wide_df.index >= min_cohort]
        if plot_cols:
            wide_df = wide_df.transpose()

        # Define line styles and colors
        num_lines = len(wide_df.columns)
        cmap = plt.get_cmap(colormap)
        colors = [cmap(1 - i / num_lines) for i in range(num_lines)]

        line_styles = ['-', '--']
        color_cycle = itertools.cycle(colors)
        line_style_cycle = itertools.cycle(line_styles)

        #plt.figure(figsize=(14, 12)) 
        for row in wide_df.columns:
            line_style = next(line_style_cycle)
            color = next(color_cycle)
            plt.plot(wide_df.index, wide_df[row], label=row, linestyle=line_style, color=color, marker='.')

        # Set title and axis labels only if provided, with increased font size
        if title:
            plt.title(title, fontsize=font_size)
        if x_label:
            plt.xlabel(x_label, fontsize=font_size)
        if y_label:
            plt.ylabel(y_label, fontsize=font_size)

        # Increase tick label font size
        plt.xticks(ticks=list(wide_df.index), rotation=45, fontsize=font_size-2)
        plt.yticks(fontsize=font_size-4)
        
        # Check if x-axis data is datetime-like
        ax = plt.gca()
        if pd.api.types.is_datetime64_any_dtype(wide_df.index):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format as YYYY-MM for datetime data
            ax.xaxis.set_major_locator(mdates.MonthLocator())  # Show monthly ticks
        else:
            # For integer-based x-axis (e.g., periods), just use default formatting
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
        
        # Format the y-axis based on the data range (in millions or thousands)
        ax = plt.gca()
        y_max = wide_df.max().max()  # Get the max value of the y-axis data

        if y_max >= 1e6:  # If the maximum value is in the millions
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x * 1e-6:.1f}M'))
        elif y_max >= 1e3:  # If the maximum value is in the thousands
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x * 1e-3:.1f}K'))

        if add_legend:
            handles, labels = plt.gca().get_legend_handles_labels()
            try:
                formatted_labels = [pd.to_datetime(label).strftime('%Y-%m') for label in labels]
            except:
                formatted_labels = labels

            title = "Cohort" if plot_cols else "Rolling Month"
            plt.legend(handles, formatted_labels, title=title, bbox_to_anchor=(1.05, 0, 0.1, 1),
                    bbox_transform=plt.gca().transAxes, borderaxespad=0, fontsize=font_size-6)

        plt.grid(True)
        plt.tight_layout()

        if show:
            plt.show()

    @staticmethod
    def boxplot(df, min_cohort=None, showfliers=True, whis=[1, 99], 
                x_label=None, y_label=None, title=None, color_palette="Set2", figsize=(12, 8), font_size=14):
        """
        Plots a boxplot for the given DataFrame with customizable options.

        Parameters:
        - df (pd.DataFrame): The DataFrame to plot.
        - min_cohort (str, optional): Minimum cohort to include based on index.
        - showfliers (bool, optional): Whether to display outliers.
        - whis (list, optional): Percentile whisker range (default [1, 99]).
        - x_label (str, optional): The label for the x-axis.
        - y_label (str, optional): The label for the y-axis.
        - title (str, optional): The title for the plot.
        - color_palette (str, optional): Color palette for the plot.
        - figsize (tuple, optional): Size of the figure (default (12, 8)).
        - font_size (int, optional): Font size for the labels and title.
        """
        if min_cohort is not None:
            df = df[df.index >= min_cohort]
        
        # Set up the figure size
        plt.figure(figsize=figsize)
        
        # Create the boxplot with the specified color palette
        sns.boxplot(data=df, palette=color_palette, whis=whis, showfliers=showfliers)

        # Customize labels and title with font sizes if provided
        if x_label:
            plt.xlabel(x_label, fontsize=font_size)
        if y_label:
            plt.ylabel(y_label, fontsize=font_size)
        if title:
            plt.title(title, fontsize=font_size + 2)

        # Rotate x-axis labels for readability
        plt.xticks(rotation=45, fontsize=font_size)

        # Add a stripplot for outliers if showfliers is enabled
        if showfliers:
            sns.stripplot(data=df, color='red', alpha=0.5, jitter=True)

        # Adjust the layout for better aesthetics
        plt.tight_layout()

        # Show the plot
        plt.show()

    @staticmethod
    def plot_wide_df_projections(wide_df, wide_df_projections, min_cohort=None, add_legend=True, plot_cols=True, colormap="viridis", show=True,
                                title=None, x_label=None, y_label=None, font_size=14):
        """
        Plots a wide-format DataFrame with projections filled in for NaN values.
        Projects the values using wide_df_projections and differentiates the original and projected lines.
        An 'X' marker is placed where projections start.

        Parameters:
        - wide_df (pd.DataFrame): The original DataFrame to plot.
        - wide_df_projections (pd.DataFrame): The DataFrame containing the projected values to fill NaNs in the original data.
        - min_cohort (str, optional): Minimum cohort to include based on index.
        - add_legend (bool, optional): Whether to add a legend.
        - plot_cols (bool, optional): Whether to plot columns as lines.
        - colormap (str, optional): Colormap for the lines.
        - show (bool, optional): Whether to display the plot immediately.
        - title (str, optional): The title of the plot. If None, no title will be displayed.
        - x_label (str, optional): The label for the x-axis. If None, no label will be displayed.
        - y_label (str, optional): The label for the y-axis. If None, no label will be displayed.
        - font_size (int, optional): Font size for title, axis labels, and ticks.
        """
        if min_cohort is not None:
            wide_df = wide_df[wide_df.index >= min_cohort]
            wide_df_projections = wide_df_projections[wide_df_projections.index >= min_cohort]

        if plot_cols:
            wide_df = wide_df.transpose()
            wide_df_projections = wide_df_projections.transpose()

        # Fill NaNs in wide_df with projections from wide_df_projections
        wide_df_filled = wide_df.combine_first(wide_df_projections)

        # If projections go beyond the original data range, ensure they are plotted
        all_dates = wide_df_filled.index.union(wide_df_projections.index)

        # Define line styles and colors for original and projected data
        num_lines = len(wide_df.columns)
        cmap = plt.get_cmap(colormap)
        original_colors = [cmap(i / num_lines) for i in range(num_lines)]  # Colors for original data

        for i, row in enumerate(wide_df.columns):
            color = original_colors[i]
            valid_data = wide_df[row].dropna()  # Drop NaN values for the original data
            
            # Check if the index is datetime-like and format accordingly
            if isinstance(row, pd.Timestamp):
                formatted_label = row.strftime('%Y-%m')  # Use the first date as the label
            else:
                formatted_label = str(row)

            # Plot original data
            plt.plot(valid_data.index, valid_data, label=formatted_label + " (Original)", linestyle='-', color=color, marker='o')

        # Plot projected data (both NaNs and beyond original data) with dashed lines and diamond markers
        for i, row in enumerate(wide_df.columns):
            projection_color = original_colors[i]
            projection_data = wide_df_projections[row].dropna()

            if not projection_data.empty:
                # Check if the row is datetime-like and format accordingly
                if isinstance(row, pd.Timestamp):
                    formatted_label = row.strftime('%Y-%m')  # Use the first date as the label
                else:
                    formatted_label = str(row)
            
                plt.plot(projection_data.index, projection_data, label=formatted_label + " (Projection)", linestyle='--', color=projection_color, marker='o')
                # Find the first point where projection starts (where original data ends)
                first_projection_idx = wide_df[row].last_valid_index()
                if first_projection_idx is not None and first_projection_idx in projection_data.index:
                    # Place an 'X' marker at the start of the projections
                    plt.plot(first_projection_idx, wide_df_projections[row].loc[first_projection_idx], marker='x', color='red', markersize=10, mew=3)


            # Set title and axis labels only if provided, with increased font size
        if title:
            plt.title(title, fontsize=font_size)
        if x_label:
            plt.xlabel(x_label, fontsize=font_size)
        if y_label:
            plt.ylabel(y_label, fontsize=font_size)

        # Increase tick label font size
        plt.xticks(ticks=list(all_dates), rotation=45, fontsize=font_size - 2)
        plt.yticks(fontsize=font_size - 4)

        # Check if x-axis data is datetime-like
        ax = plt.gca()
        if pd.api.types.is_datetime64_any_dtype(all_dates):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format as YYYY-MM for datetime data
            ax.xaxis.set_major_locator(mdates.MonthLocator())  # Show monthly ticks
        else:
            # For integer-based x-axis (e.g., periods), just use default formatting
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

        # Format the y-axis based on the data range (in millions or thousands)
        y_max = wide_df_filled.max().max()  # Get the max value of the y-axis data

        if y_max >= 1e6:  # If the maximum value is in the millions
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x * 1e-6:.1f}M'))
        elif y_max >= 1e3:  # If the maximum value is in the thousands
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x * 1e-3:.1f}K'))

        if add_legend:
            handles, labels = plt.gca().get_legend_handles_labels()
            try:
                formatted_labels = [pd.to_datetime(label).strftime('%Y-%m') for label in labels]
            except:
                formatted_labels = labels

            title = "Cohort" if plot_cols else "Rolling Month"
            plt.legend(handles, formatted_labels, title=title, bbox_to_anchor=(1.05, 0, 0.1, 1),
                    bbox_transform=plt.gca().transAxes, borderaxespad=0, fontsize=font_size-6)

        plt.grid(True)
        plt.tight_layout()

        if show:
            plt.show()