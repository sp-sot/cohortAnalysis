
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
    
class RawDataPreprocessor: 
    def __init__(self, raw_customer_payment_df, spend_df):
        self.payment_df = raw_customer_payment_df
        self.spend_df = spend_df

    def get_payment_data(self):
        return self.payment_df
    
    def get_spend_data(self):
        return self.spend_df
    
    def check_monthly_payment_gaps(self):
        monthly_data = self.payment_df[self.payment_df["payment_frequency"] == "monthly"]
        return RawDataPreprocessor.check_payment_gaps(monthly_data, 1)
    
    def check_quarterly_payment_gaps(self):
        quarterly_data = self.payment_df[self.payment_df["payment_frequency"] == "quarterly"]
        return RawDataPreprocessor.check_payment_gaps(quarterly_data, 3)

    def get_number_of_payments_per_custoner(self):
        return self.payment_df['customer_id'].value_counts().reset_index().rename(columns={'index': 'customer_id', 'customer_id': 'num_payments'})
    
    def get_max_payment_date(self):
        return self.payment_df["payment_date"].max()
    
    def get_min_payment_date(self):
        return self.payment_df["payment_date"].min()
    
    def get_payment_frequency_count_per_customer(self):
        return self.payment_df.groupby('customer_id')['payment_frequency'].nunique().reset_index().rename(columns={'payment_frequency': 'num_payment_frequencies'})
    
    def get_unique_payment_date_days(self):
        return self.payment_df['payment_date'].dt.day.unique()
    
    def remove_customers_with_only_zero_payments(self):
        customers_with_only_zero_payments = self.payment_df.groupby('customer_id')['amount'].sum()
        customers_to_remove = customers_with_only_zero_payments[customers_with_only_zero_payments == 0].index
        df_filtered = self.payment_df[~self.payment_df['customer_id'].isin(customers_to_remove)]
        self.payment_df = df_filtered   

    def add_cohort(self):
        df_sorted = self.payment_df[self.payment_df["amount"] > 0].sort_values(by=["customer_id", "payment_date"])
        first_nonzero_payment = df_sorted.drop_duplicates(subset='customer_id', keep='first')[['customer_id', 'payment_date']]
        first_nonzero_payment = first_nonzero_payment.rename(columns={'payment_date': 'first_nonzero_payment'})
        self.payment_df = self.payment_df.merge(first_nonzero_payment, on='customer_id', how='left')
        self.payment_df["cohort"] = self.payment_df['first_nonzero_payment'].values.astype('datetime64[M]')
        self.payment_df = self.payment_df.drop(columns=['first_nonzero_payment'])

    def add_payment_period(self):
        self.payment_df["payment_period"] = self.payment_df["payment_date"].dt.to_period('M').astype(int) - self.payment_df["cohort"].dt.to_period('M').astype(int)

    def remove_incomplete_month_data(self, year, month):
        period_to_remove = pd.Period(year=year, month=month, freq='M')
        self.payment_df =  self.payment_df[self.payment_df["payment_date"].dt.to_period('M') != period_to_remove]
    
    def remove_outlier_customer(self, customer_id):
        self.payment_df = self.payment_df[self.payment_df["customer_id"] != customer_id]
    
    def remove_incomplete_cohort_from_spend_data(self, cohort):
        self.spend_df = self.spend_df[self.spend_df["cohort"] != cohort]
    
    @staticmethod
    def check_payment_gaps(df, payment_period_distance):
        """
        Check if there are any gaps in payment periods for each customer
        payment_period_distance: int, the maximum gap between consecutive payment periods    
        """
        # Sort the DataFrame by customer_id and payment_period
        df = df.sort_values(by=['customer_id', 'payment_period'])
        # Calculate the difference between consecutive payment periods within each customer group
        df['period_diff'] = df.groupby('customer_id')['payment_period'].diff()
        # Check if any difference is greater than payment_period_distance, which indicates a gap
        df['has_gap'] = df['period_diff'] > payment_period_distance
        # Group by customer_id and check if any gaps exist
        result = df.groupby('customer_id')['has_gap'].max().reset_index()
        # If has_gap is True, that customer has a gap in payment periods
        result['has_gaps'] = result['has_gap'].fillna(False)
        # Drop intermediate columns
        return result[['customer_id', 'has_gaps']]