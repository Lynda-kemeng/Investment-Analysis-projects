
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def process_and_visualize_pe_ratios(price_return_path, eps_book_path, company_info_path, output_summary_main, output_summary_gem):
    price_data = pd.read_csv(price_return_path)
    eps_book_data = pd.read_csv(eps_book_path)
    company_data = pd.read_csv(company_info_path)

    eps_book_data = eps_book_data[eps_book_data['Typrep'] != 'B'].drop(columns=['Typrep', 'ShortName_EN'])
    price_data.columns = ['stock', 'record_date', 'closing_price', 'market_cap', 'monthly_return']
    eps_book_data.columns = ['stock', 'report_date', 'eps', 'book_value']
    company_data = company_data.drop(columns=['Stknme_en', 'Listdt', 'Estbdt'])
    company_data.columns = ['stock', 'market']


    price_data['record_date'] = pd.to_datetime(price_data['record_date']).dt.to_period('M')
    eps_book_data['report_date'] = pd.to_datetime(eps_book_data['report_date']).dt.to_period('M')
    eps_book_data['report_date'] = (eps_book_data['report_date'].dt.to_timestamp() + pd.DateOffset(months=1)).dt.to_period('M')

    combined_data = pd.merge(price_data, eps_book_data, left_on=['stock', 'record_date'], right_on=['stock', 'report_date'], how='left').drop(columns=['report_date'])


    combined_data['eps'] = combined_data.groupby('stock')['eps'].ffill()
    combined_data['book_value'] = combined_data.groupby('stock')['book_value'].ffill()
    combined_data['pe_ratio'] = combined_data['closing_price'] / combined_data['eps']
    combined_data['pb_ratio'] = combined_data['closing_price'] / combined_data['book_value']
    combined_data.replace([np.inf, -np.inf], np.nan, inplace=True)  # 关键修改[[9, 15]]

    combined_data = pd.merge(combined_data, company_data, on=['stock'], how='left')
    combined_data['record_date'] = combined_data['record_date'].dt.to_timestamp()
    combined_data['market'] = combined_data['market'].replace([1, 4, 64], "Main Board").replace([16, 32], "GEM")

    # Grouping and Summary Statistics
    pe_by_market = combined_data.groupby(['market', 'record_date'])['pe_ratio'].median().unstack(level=0)
    main_summary = combined_data[combined_data['market'] == 'Main Board'].describe().round(2)
    gem_summary = combined_data[combined_data['market'] == 'GEM'].describe().round(2)

    # Output Summary Statistics
    main_summary.to_csv(output_summary_main)
    gem_summary.to_csv(output_summary_gem)

    # Plotting
    pe_by_market.plot(figsize=(16, 7), marker='o')
    plt.title('Median P/E Ratios Across Market Types')
    plt.xlabel('Time')
    plt.ylabel('Median P/E Ratio')
    plt.grid(axis='y', linestyle='--')
    plt.legend(title='Market')
    plt.show()

# Execution
process_and_visualize_pe_ratios(
    "E:\Desktop\data\TRD_Mnth.csv",
    "E:\Desktop\data\FI_T9.csv",
    "E:\Desktop\data\TRD_Co.csv",
    'summary_main1.csv',
    'summary_gem1.csv'
)

