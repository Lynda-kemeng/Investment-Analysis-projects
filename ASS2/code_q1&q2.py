import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings


price_data = pd.read_csv("E:\\Desktop\\data\\TRD_Mnth.csv")
volatility_data = pd.read_csv("E:\\Desktop\\data\\STK_MKT_STKBTAL.csv")
financials = pd.read_csv("E:\\Desktop\\data\\FI_T5.csv")

price_data.rename(columns={
    'Stkcd': 'firm_id',
    'Trdmnt': 'report_date',
    'Mclsprc': 'closing_price',
    'Mretnd': 'monthly_ret'
}, inplace=True)

volatility_data.rename(columns={
    'Symbol': 'firm_id',
    'TradingDate': 'report_date',
    'Volatility': 'volatility'
}, inplace=True)


financials.rename(columns={
    'FI_T5.Stkcd': 'firm_id',
    'FI_T5.Accper': 'report_date',
    'FI_T5.Typrep': 'report_type',
    'FI_T9.F091001A': 'book_value',
    'FI_T5.F050504C': 'return_on_equity'
},inplace=True)

financials = financials[financials['report_type'] != 'B']
financials.drop(columns=['report_type'], inplace=True)

financials.loc[financials['report_date'] == '2009-09-30', 'report_date'] = '2009-11-01'

price_data['report_date'] = pd.to_datetime(price_data['report_date']).dt.to_period('M')
financials['report_date'] = pd.to_datetime(financials['report_date']).dt.to_period('M')
financials['report_date'] = (financials['report_date'].dt.to_timestamp() + pd.DateOffset(months=1)).dt.to_period('M')
volatility_data['report_date'] = pd.to_datetime(volatility_data['report_date']).dt.to_period('M')

merged_data = pd.merge(price_data, financials, on=['firm_id', 'report_date'], how='left')
merged_data = pd.merge(merged_data, volatility_data, on=['firm_id', 'report_date'], how='left')

merged_data['return_on_equity'] = merged_data.groupby('firm_id')['return_on_equity'].ffill()
merged_data['book_value'] = merged_data.groupby('firm_id')['book_value'].ffill()

merged_data['price_to_book'] = merged_data['closing_price'] / merged_data['book_value']

pb_5th = merged_data['price_to_book'].quantile(0.05)
pb_95th = merged_data['price_to_book'].quantile(0.95)
merged_data = merged_data[(merged_data['price_to_book'] > pb_5th) & (merged_data['price_to_book'] < pb_95th)]


merged_data.rename(columns={'report_date': 'month'}, inplace=True)

data_2009 = merged_data[merged_data['month'] == '2009-12']
merged_data = merged_data[merged_data['month'] != '2009-12']

data_2010Q4 = merged_data[merged_data['month'] == '2010-12']
data_2010Q4.dropna(subset=['price_to_book', 'return_on_equity', 'volatility'], inplace=True)

Y = data_2010Q4['price_to_book']
X = data_2010Q4[['return_on_equity', 'volatility']]
X = sm.add_constant(X)

model = sm.OLS(Y, X)
results = model.fit()

print(results.summary())


with open("E:\\Desktop\\data\\q1_results.txt", 'w') as file:
    file.write(results.summary().as_text())

warnings.filterwarnings("ignore", category=DeprecationWarning)
merged_data = pd.concat([merged_data, data_2009])

merged_data = merged_data.dropna(subset=['price_to_book'])
merged_data['prev_month_pb'] = merged_data.groupby('firm_id')['price_to_book'].shift(1)
merged_data = merged_data.dropna(subset=['prev_month_pb'])

sorted_data = merged_data.groupby('month').apply(lambda x: x.sort_values('prev_month_pb')).reset_index(drop=True)

def assign_deciles(group):
    group.sort_values('prev_month_pb', inplace=True)
    return group

sorted_data = sorted_data.groupby('month').apply(assign_deciles).reset_index(drop=True)
sorted_data['pb_decile'] = sorted_data.groupby('month')['prev_month_pb'].transform(lambda x: pd.qcut(x, 10, labels=False))
sorted_data['pb_decile'] += 1

decile_avg_return = sorted_data.groupby(['month', 'pb_decile'])['monthly_ret'].mean().reset_index()
decile_return_all = decile_avg_return.groupby('pb_decile')['monthly_ret'].mean().reset_index()


plt.bar(decile_return_all['pb_decile'], decile_return_all['monthly_ret'], color='steelblue', alpha=0.8)
plt.xlabel('Decile')
plt.ylabel('Average Monthly Return')
plt.title('Average Monthly Return by P/B Decile')
plt.xticks(range(1, 11))
plt.show()


