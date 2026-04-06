import pandas as pd

clean_data = pd.read_csv('data/jd_sn_tractors.csv')

clean_data['serial'] = clean_data['serial'].str.replace(r'[ -]', '', regex=True)

clean_data.to_csv('data/cleaned_data.csv', index=False, header=True)

clean_data.head(200).to_csv('data/cleaned_data_subset.csv', index=False, header=True)