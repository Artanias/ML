import pickle
import pandas as pd
import matplotlib.pyplot as plt

path_pickles = "./Models/"

list_pickles = [
    # "df_models_gbc.pickle",
    "df_models_knnc.pickle",
    "df_models_lrc.pickle",
    "df_models_mnbc.pickle",
    "df_models_rfc.pickle",
    "df_models_svc.pickle"
]

df_summary = pd.DataFrame()

for pickle_ in list_pickles:

    path = path_pickles + pickle_

    with open(path, 'rb') as data:
        df = pickle.load(data)

    df_summary = df_summary.append(df)

df_summary = df_summary.reset_index().drop('index', axis=1)

print(df_summary.sort_values('Test Set Accuracy', ascending=False))
print(df_summary['Test Set Accuracy'])

use, ax = plt.subplots(1, 1, figsize=(12, 5))
ax.bar(df_summary['Model'], df_summary['Test Set Accuracy'])
ax.set_xlabel('Models')
ax.set_ylabel('Accuracy')
ax.set_title('Test Accuracy')
plt.savefig('./Grafics/BMS.png')
plt.show()
