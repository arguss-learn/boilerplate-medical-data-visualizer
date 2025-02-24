import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import the data from medical_examination.csv
df = pd.read_csv('medical_examination.csv')

# 2. Add an overweight column
df['height_m'] = df['height'] / 100  # Convert height from cm to meters
df['BMI'] = df['weight'] / (df['height_m'] ** 2)  # Calculate BMI
df['overweight'] = (df['BMI'] > 25).astype(int)  # 1 if BMI > 25, else 0

# Remove temporary columns
df.drop(columns=['height_m', 'BMI'], inplace=True)

# 3. Normalize cholesterol and gluc (0 = good, 1 = bad)
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4. Draw the Categorical Plot
def draw_cat_plot():
    # 5. Melt the DataFrame
    df_cat = pd.melt(df, id_vars='cardio', 
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6. Create a catplot with specified order
    order = ['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']
    fig = sns.catplot(data=df_cat, x='variable', hue='value', col='cardio', kind='count', height=5, aspect=0.8, order=order)

    # 7. Label the plot
    fig.set_axis_labels('variable', 'total')
    fig.set_titles('Cardio: {col_name}')
    fig.fig.suptitle('Count of Variables by Cardio Outcome')

    # 8. Save figure
    fig.savefig('catplot.png')
    return fig.fig


# 9. Draw the Heat Map
def draw_heat_map():
    # 10. Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 11. Calculate the correlation matrix
    corr = df_heat.corr().round(1)

    # 12. Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 13. Plot the heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap='coolwarm', square=True, cbar_kws={"shrink": 0.8}, ax=ax)

    # 14. Title and save
    plt.title('Correlation Heatmap')
    fig.savefig('heatmap.png')
    return fig
