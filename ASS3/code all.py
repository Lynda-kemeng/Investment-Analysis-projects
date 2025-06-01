
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import skew, kurtosis, shapiro


index_path = "E:\\Desktop\\data\\TRD_Index.csv"
df_idx = pd.read_csv(index_path)
df_idx.columns = ['ticker', 'date', 'price']

df_idx['date'] = pd.to_datetime(df_idx['date']).dt.to_period('M')


df_idx = df_idx.sort_values(['date', 'ticker'])
df_idx = df_idx.groupby(['date', 'ticker'], as_index=False).agg({'price': 'last'})


df_idx['ticker'] = df_idx['ticker'].astype(str)
df_idx = df_idx[df_idx['ticker'] == '300']

df_idx['monthly_return'] = df_idx['price'].pct_change()
df_idx = df_idx[df_idx['date'] != '2006-01']

# 计算描述统计量
mean_ret = df_idx['monthly_return'].mean()
std_ret = df_idx['monthly_return'].std()
skew_ret = skew(df_idx['monthly_return'])
kurt_ret = kurtosis(df_idx['monthly_return'])

desc_stats = pd.DataFrame({
    'Mean': [mean_ret],
    'Standard deviation': [std_ret],
    'Skewness': [skew_ret],
    'Kurtosis': [kurt_ret]
}).round(10)

print(desc_stats)
# desc_stats.to_csv('output/index_stats.csv', index=False)


plt.hist(df_idx['monthly_return'], bins=40, range=(-0.3, 0.3), density=True, edgecolor='black')
plt.title('Index Monthly Returns')
plt.xlabel('Return')
plt.ylabel('Frequency')
plt.show()


w_stat, p_val = shapiro(df_idx['monthly_return'])
print(f"Shapiro-Wilk test statistic: {w_stat}")
print(f"p-value: {p_val}")

shapiro_stats = pd.DataFrame({
    'Shapiro-Wilk test statistic': [w_stat],
    'p-value': [p_val]
}).round(6)
shapiro_stats.to_csv("E:\\Desktop\\data\\test_stats.csv", index=False)

week_path1 = "E:\\Desktop\\data\\1\\TRD_Week.csv"
week_path2 = "E:\\Desktop\\data\\1\\TRD_Week1.csv"
df_week1 = pd.read_csv(week_path1)
df_week2 = pd.read_csv(week_path2)

df_stock = pd.concat([df_week1, df_week2], axis=0)
df_stock.columns = ['ticker', 'week', 'stock_return', 'category']


df_stock = df_stock[df_stock['category'].isin([1, 4, 64])].drop(columns=['category'])

market_return = df_stock.groupby('week')['stock_return'].mean().reset_index()
market_return.columns = ['week', 'market_return']


rf_path = "E:\\Desktop\\data\\weekly_risk_free_rate.xlsx"
df_rf = pd.read_excel(rf_path)
df_rf.columns = ['week', 'rf_return']
df_rf['week'] = df_rf['week'].dt.strftime('%Y-%W').astype(str)
df_rf[['year', 'wk_num']] = df_rf['week'].str.split('-', expand=True)
df_rf['wk_num'] = (df_rf['wk_num'].astype(int) + 1).astype(str).str.zfill(2)
df_rf['week'] = df_rf['year'] + '-' + df_rf['wk_num']
df_rf.drop(columns=['year', 'wk_num'], inplace=True)


df_stock = pd.merge(df_stock, market_return, on='week', how='left')
df_stock.dropna(inplace=True)
df_stock['week'] = df_stock['week'].astype(str)


reg_outcomes = pd.DataFrame()
df_stock_sub = df_stock[df_stock['week'].str.startswith(('2017', '2018'))]


for tic in df_stock_sub['ticker'].unique():
    sub_data = df_stock_sub[df_stock_sub['ticker'] == tic]
    X = sm.add_constant(sub_data['market_return'])
    y = sub_data['stock_return']
    model = sm.OLS(y, X).fit()
    try:
        alpha_val = model.params['const']
        beta_val = model.params['market_return']
        r2_val = model.rsquared
        reg_outcomes = pd.concat(
            [reg_outcomes, pd.DataFrame({'ticker': [tic], 'alpha': [alpha_val], 'beta': [beta_val], 'r_squared': [r2_val]})],
            ignore_index=True
        )
    except KeyError:
        continue

reg_outcomes = reg_outcomes[reg_outcomes['r_squared'] != 1]
reg_outcomes = reg_outcomes.sort_values(by='beta')
reg_outcomes['group'] = pd.qcut(reg_outcomes['beta'], 10, labels=False)
reg_outcomes['group'] = (reg_outcomes['group'] + 1).astype(int)

# 将分组信息合并回股票数据
df_stock = pd.merge(df_stock, reg_outcomes[['ticker', 'group']], on='ticker', how='left')


portfolios = df_stock.groupby(['week', 'group'])['stock_return'].mean().reset_index()
portfolios.columns = ['week', 'group', 'port_return']

portfolios = pd.merge(portfolios, df_rf, on='week', how='left')
portfolios = pd.merge(portfolios, market_return, on='week', how='left')
portfolios.dropna(inplace=True)
portfolios['excess_return'] = portfolios['port_return'] - portfolios['rf_return']
portfolios['risk_premium'] = portfolios['market_return'] - portfolios['rf_return']


group_reg_results = pd.DataFrame()
port_sub = portfolios[portfolios['week'].str.startswith(('2019', '2020'))]

for grp in port_sub['group'].unique():
    grp_data = port_sub[port_sub['group'] == grp]
    X_grp = sm.add_constant(grp_data['risk_premium'])
    y_grp = grp_data['excess_return']
    reg_model = sm.OLS(y_grp, X_grp).fit()
    alpha_grp = reg_model.params['const']
    beta_grp = reg_model.params['risk_premium']
    t_alpha = reg_model.tvalues.iloc[0]
    t_beta = reg_model.tvalues.iloc[1]
    p_alpha = reg_model.pvalues.iloc[0]
    p_beta = reg_model.pvalues.iloc[1]
    r2_grp = reg_model.rsquared
    group_reg_results = pd.concat(
        [group_reg_results, pd.DataFrame({'group': [grp],
                                          'alpha': [alpha_grp], 'alpha_t': [t_alpha], 'alpha_p': [p_alpha],
                                          'beta': [beta_grp], 'beta_t': [t_beta], 'beta_p': [p_beta],
                                          'r_squared': [r2_grp]})],
        ignore_index=True
    )
group_reg_results.to_csv("E:\\Desktop\\data\\reg_2_result.csv", index=False)
port_2021 = portfolios[portfolios['week'].str.startswith(('2021', '2022'))]
port_2021 = port_2021.groupby('group')['excess_return'].mean().reset_index()
port_2021 = pd.merge(port_2021, group_reg_results[['group', 'beta']], on='group', how='left')



X_sec = sm.add_constant(port_2021['beta'])
y_sec = port_2021['excess_return']
sec_model = sm.OLS(y_sec, X_sec).fit()
print(sec_model.summary())


final_results = pd.DataFrame({
    'gamma_0': [sec_model.params['const']],
    'gamma_1': [sec_model.params['beta']],
    'gamma_0_t': [sec_model.tvalues[0]],
    'gamma_1_t': [sec_model.tvalues[1]],
    'R^2': [sec_model.rsquared],
    'F-statistic': [sec_model.fvalue],
    'P': [sec_model.f_pvalue]
}).round(5)

final_results.to_csv("E:\\Desktop\\data\\Table_3_result.csv", index=False)


plt.scatter(port_2021['beta'], port_2021['excess_return'])
plt.plot(port_2021['beta'],
         sec_model.params['const'] + sec_model.params['beta'] * port_2021['beta'],
         color='red')
plt.title('Regression Results')
plt.xlabel('Beta')
plt.ylabel('Excess Return')
plt.show()
