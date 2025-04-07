import pandas as pd
from matplotlib import pyplot as plt


def analyze_metrics_consistency(df, start_year=2010, end_year=2020):

    df.columns = ['cid', 'date', 'revenue', 'roe']

    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year

    yr_range = range(start_year, end_year + 1)
    valid_years_set = set(yr_range)


    clean_df = df.dropna(subset=['roe', 'revenue']).query('year in @yr_range')

    complete_cids = clean_df.groupby('cid').filter(
        lambda g: set(g['year']) == valid_years_set
    )['cid'].unique()


    analysis_df = clean_df[clean_df['cid'].isin(complete_cids)].sort_values(
        ['cid', 'year']
    )


    analysis_df['revenue_growth'] = analysis_df.groupby('cid')['revenue'].transform(
        lambda s: s.pct_change().fillna(0) * 100
    )

    years_tracked = list(yr_range)
    base_companies = analysis_df['cid'].unique()
    total_companies = len(base_companies)

    roe_consistent = set(base_companies)
    growth_consistent = set(base_companies)

    roe_perc = [50.0]
    growth_perc = [50.0]

    for idx, current_yr in enumerate(years_tracked[1:]):

        roe_cutoff = analysis_df[analysis_df['year'] == current_yr]['roe'].median()

        growth_cutoff = analysis_df[analysis_df['year'] == current_yr]['revenue_growth'].median()

        yr_roe = analysis_df[(analysis_df['year'] == current_yr) &
                             (analysis_df['roe'] > roe_cutoff)]['cid'].unique()
        yr_growth = analysis_df[(analysis_df['year'] == current_yr) &
                                (analysis_df['revenue_growth'] > growth_cutoff)]['cid'].unique()

        roe_consistent &= set(yr_roe)
        growth_consistent &= set(yr_growth)

        roe_perc.append(len(roe_consistent) / total_companies * 100)
        growth_perc.append(len(growth_consistent) / total_companies * 100)

    return years_tracked, roe_perc, growth_perc

df = pd.read_csv(r"E:\Desktop\data\problem 3_data.csv")
x_vals, roe_data, growth_data = analyze_metrics_consistency(df)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, roe_data,
         marker='D', linestyle='-', linewidth=2,
         color='#4B0082',
         label='Percentage of Firms Consistently Above Median ROE')

plt.plot(x_vals, growth_data,
         marker='^', linestyle=':', linewidth=2.5,
         color='#DAA520',
         label='Percentage of Firms Consistently Above Median Revenue Growth')

plt.grid(True, alpha=0.3)
plt.xticks(range(2010, 2021, 2))
plt.ylim(0, 55)
plt.legend(loc='upper right', frameon=False)

plt.suptitle('Financial Performance Consistency Over Years', y=0.97, size=14)
plt.xlabel('Year', fontsize=10, labelpad=8)
plt.ylabel('Percentage of Companies', fontsize=10, labelpad=8)

plt.tight_layout()
plt.show()
