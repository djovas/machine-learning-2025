# %%

import pandas as pd

df = pd.read_excel('data/dados_cerveja.xlsx')
df.head()

# %%

features = ['temperatura', 'copo', 'espuma', 'cor']
target = 'classe'

X = df[features]
y = df[target]

# replacing the the strings to numbers because the sklearn use numbers

X = X.replace({
    "mud": 1, "pint": 0,
    "sim": 1, "não": 0,
    "clara": 1, "escura": 0
})

X

# %%

from sklearn import tree

model = tree.DecisionTreeClassifier()

model.fit( X=X, y=y )


# %%

import matplotlib.pyplot as plt

plt.figure(dpi=400)

tree.plot_tree(model, 
    feature_names=features, 
    class_names=model.classes_, 
    filled=True
    )

# %%
