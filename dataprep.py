import pandas as pd
df=pd.read_csv("news_summary.csv",encoding = "ISO-8859-1")
columns=df.columns
new_df=df[['ctext','text']]
#print(df.describe())
#print(new_df.head())
#print("Description:",new_df['ctext'][0])
#print("Summary:",new_df['text'][0])
new_df.to_csv("summary.csv",encoding = "ISO-8859-1")
