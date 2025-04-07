
import pandas as pd

data_paths = {
    'price': r"E:\Desktop\data\TRD_Mnth.csv",
    'eps': r'E:\Desktop\data\FI_T9.csv',
    'rnd': r'E:\Desktop\data\FS_Comins.csv',
    'roa_roe': r'E:\Desktop\data\FI_T5.csv',
    'info': r'E:\Desktop\data\TRD_Co.csv',
    'balance': r'E:\Desktop\data\FS_Combas.csv',
    'quarters': r'E:\Desktop\data\quarter_prepare.csv'
}


datasets = {k: pd.read_csv(v) for k, v in data_paths.items()}


def clean_data(df):
    if 'Typrep' in df.columns:
        df = df.query("Typrep != 'B'").drop(columns=['Typrep'])
    return df

for key in ['balance', 'eps', 'rnd', 'roa_roe']:
    datasets[key] = clean_data(datasets[key])

column_settings = {
    'rnd': {
        'drop': ['ShortName_EN'],
        'columns': ['stock_code', 'date', 'R&D_expense']
    },
    'balance': {
        'drop': ['ShortName_EN'],
        'columns': ['stock_code', 'date', 'total_asset', 'total_liability']
    },
    'info': {
        'drop': ['Stknme_en', 'Listdt'],
        'columns': ['stock_code', 'est_date', 'market_type']
    },
    'roa_roe': {
        'drop': ['ShortName_EN'],
        'columns': ['stock_code', 'date', 'ROA', 'ROE']
    }
}

# 批量处理列操作
for ds in ['rnd', 'balance', 'info', 'roa_roe']:
    datasets[ds] = (datasets[ds]
        .drop(columns=column_settings[ds]['drop'])
        .set_axis(column_settings[ds]['columns'], axis=1))

period_config = [
    (datasets['rnd'], 'date', 'Q'),
    (datasets['balance'], 'date', 'Q'),
    (datasets['info'], 'est_date', 'D'),
    (datasets['roa_roe'], 'date', 'Q'),
    (datasets['quarters'], 'date', 'Q-DEC')
]

for df, col, freq in period_config:
    if freq == 'Q-DEC':
        df[col] = df[col].astype(f'period[{freq}]')
    else:
        df[col] = pd.to_datetime(df[col]).dt.to_period(freq)

merge_sequence = [
    (datasets['balance'], ['date'], 'left'),
    (datasets['rnd'], ['stock_code', 'date'], 'left'),
    (datasets['info'], ['stock_code'], 'left'),
    (datasets['roa_roe'], ['stock_code', 'date'], 'left')
]

merged_data = datasets['quarters'].copy()
for dataset, on_cols, how in merge_sequence:
    merged_data = merged_data.merge(
        dataset,
        how=how,
        on=on_cols,
        suffixes=('', '_DROP')
    ).filter(regex='^(?!.*_DROP)')


merged_data = merged_data.assign(
    R_D_ratio = lambda x: x['R&D_expense'] / x['total_asset'],
    firm_age = lambda x: (
        x['date'].dt.end_time - x['est_date'].dt.to_timestamp()
    ).dt.days / 365
)

market_map = {
    **{k: 'Main Board' for k in [1, 4, 64]},
    **{k: 'GEM' for k in [16, 32]}
}
merged_data['market_type'] = merged_data['market_type'].map(market_map)


def generate_stats(df, market):
    return (df
        .query(f"market_type == '{market}'")
        .describe(percentiles=[.25, .5, .75])
        .round(2)
        [['R_D_ratio', 'firm_age', 'ROA', 'ROE']]
    )

for market, suffix in [('Main Board', 'main'), ('GEM', 'gem')]:
    generate_stats(merged_data, market).to_csv(f'summary2_{suffix}.csv')