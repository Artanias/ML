import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

sns.set_style("whitegrid")

df_path = './'
df_path2 = df_path + 'News_dataset.csv'
df = pd.read_csv(df_path2, sep=';')
print(df.head())

bars = alt.Chart(df).mark_bar(size=20).encode(
    x=alt.X("Category"),
    y=alt.Y("count():Q", axis=alt.Axis(title='Number of articles')),
    tooltip=[alt.Tooltip('count()', title='Number of articles'), 'Category'],
    color='Category'

)
text = bars.mark_text(
    align='center',
    baseline='bottom',
).encode(
    text='count()'
)
(bars + text).interactive().properties(
    height=300,
    width=700,
    title="Number of articles in each category",
)
(bars + text).show()

df['News_length'] = df['Content'].str.len()
plt.figure(figsize=(12.8, 6))
sns.distplot(df['News_length']).set_title('News length distribution')
df['News_length'].describe()
plt.savefig('./Grafics/Length_doc1.png')
plt.show()

quantile_95 = df['News_length'].quantile(0.95)
df_95 = df[df['News_length'] < quantile_95]
plt.figure(figsize=(12.8, 6))
sns.distplot(df_95['News_length']).set_title('News length distribution')
plt.savefig('./Grafics/Length_doc2.png')
plt.show()

plt.figure(figsize=(12.8, 6))
sns.boxplot(data=df, x='Category', y='News_length', width=.5)
plt.savefig('./Grafics/Length_in_Categorys1.png')
plt.show()

plt.figure(figsize=(12.8, 6))
sns.boxplot(data=df_95, x='Category', y='News_length')
plt.savefig('./Grafics/Length_in_Categorys2.png')
plt.show()

with open('News_dataset.pickle', 'wb') as output:
    pickle.dump(df, output)
