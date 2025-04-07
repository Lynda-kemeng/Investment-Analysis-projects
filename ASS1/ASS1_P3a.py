
import pandas as pd
from datetime import datetime as dt

raw_data = pd.read_csv("E:\Desktop\data\problem 3_data.csv")
raw_data.columns = ['symbol', 'time', 'income', 'return_on_equity']
raw_data['time'] = pd.to_datetime(raw_data['time'])

raw_data['report_year'] = raw_data['time'].apply(lambda x: x.year)
filtered_data = raw_data.query('2011 <= report_year <= 2020').copy()


def compute_annual_metrics(df):
    roe_stats = df.groupby('report_year', as_index=False).agg(
        median_ROE=('return_on_equity', 'median')
    )

    df['filled_income'] = df['income'].ffill()
    df['income_growth_rate'] = df['filled_income'].pct_change(fill_method=None)

    growth_stats = df.groupby('report_year').agg(
        median_growth=('income_growth_rate', 'median')
    ).reset_index()

    return roe_stats.merge(growth_stats, on='report_year')

annual_medians = compute_annual_metrics(filtered_data)
annual_medians = annual_medians.round({'median_ROE': 4, 'median_growth': 4})
annual_medians.to_csv('financial_metrics_report.csv', index=False)



