import numpy as np
import pandas as pd

# Ensure that all columns are displayed without truncation
pd.set_option('display.max_columns', None)

# Load the data obtained from web crawling
data = pd.read_csv("data.csv")

# Drop rows where the 'position' column has values 1, 2, or 3 (excluding bottom duo positions)
values_to_drop = [1, 2, 3]
column_to_check = 'position'
df_filtered = data[~data[column_to_check].isin(values_to_drop)]

# Create an empty DataFrame to store the final merged data
final_merged_df = pd.DataFrame()

# Iterate through the DataFrame in chunks of 4 rows
for i in range(0, len(df_filtered), 4):
    subset_df = df_filtered.iloc[i:i + 4]  # Extract 4 rows at a time
    cond_04 = (subset_df['team'] == 0) & (subset_df['position'] == 4)
    cond_05 = (subset_df['team'] == 0) & (subset_df['position'] == 5)
    cond_14 = (subset_df['team'] == 1) & (subset_df['position'] == 4)
    cond_15 = (subset_df['team'] == 1) & (subset_df['position'] == 5)

    # Extract 'id' values for identifying the rows
    a = subset_df.loc[cond_04, 'id'].tolist()

    # Append team and position suffix to column names, and add 'id' back as a column
    df_04 = subset_df[cond_04].drop(columns='id').add_suffix('_04')
    df_04['id'] = a
    df_05 = subset_df[cond_05].drop(columns='id').add_suffix('_05')
    df_05['id'] = a
    df_14 = subset_df[cond_14].drop(columns='id').add_suffix('_14')
    df_14['id'] = a
    df_15 = subset_df[cond_15].drop(columns='id').add_suffix('_15')
    df_15['id'] = a

    # Merge the four DataFrames on the 'id' column
    merged_df = pd.merge(pd.merge(pd.merge(df_04, df_05, on='id', how='outer'), df_14, on='id', how='outer'), df_15,
                         on='id', how='outer')

    # Concatenate the merged DataFrame to the final DataFrame
    final_merged_df = pd.concat([final_merged_df, merged_df], ignore_index=True)

# Remove rows where 'kill_influence' values exceed 1
final_merged_df = final_merged_df[
    (final_merged_df['kill_influence_04'] <= 1) & (final_merged_df['kill_influence_05'] <= 1) &
    (final_merged_df['kill_influence_14'] <= 1) & (final_merged_df['kill_influence_15'] <= 1)]

# Remove duplicate rows based on the 'id' column
final_merged_df = final_merged_df.drop_duplicates(subset='id', keep='first')

# Display summary statistics for the final DataFrame
print(final_merged_df.describe())

# Save the merged data to a CSV file
save_path = "merged_data_name.csv"
final_merged_df.to_csv(save_path, index=False)

print(f"The CSV file has been successfully saved to {save_path}.")
