https://colab.research.google.com/notebooks/intro.ipynb
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
import matplotlib 
matplotlib.rcParams["figure.figsize"] = (20,10)

df1 = pd.read_excel("/content/House_Rent_Train.xlsx")
df1.head()

df1.shape

df1.columns

df1['bhk'].unique()

df1['bhk'] = df1['bhk'].str.extract('(\d+)').astype(float)

df1['bhk'].value_counts()

df2 = df1.drop(['activation_date', 'latitude', 'longitude', 'floor'],axis='columns')
df2.shape

df1.head()

df2.isnull().sum()

df3 = df2.dropna()
df3.isnull().sum()

df3.shape

values_to_drop = ['bhk2', 'bhk3', '1BHK1','BHK4PLUS']
df4 = df3[~df3['bhk'].isin(values_to_drop)]

df4['bhk'].unique()

df4['bhk'].value_counts()

df4['balconies'].unique()

df4['balconies'].value_counts()

values_to_keep = [1, 2, 3, 4]
df5 = df4[df4['balconies'].isin(values_to_keep)]

df5['balconies'].unique()

df5['cup_board'].unique()

df5['cup_board'].value_counts()

df6 = df5[df5['cup_board'] <= 7]


df6['cup_board'].value_counts()

df6['bathroom'].value_counts()

df7 = df6[df6['bathroom'] <= 5]


df7['bathroom'].value_counts()

df7['property_age'].value_counts()

df8 = df7[df7['property_age'] > -1]

counts = df8['property_age'].value_counts()
to_keep = counts[counts > 9].index
df9 = df8[df8['property_age'].isin(to_keep)]

df9['property_age'].value_counts()


df9.shape

len(df9.locality.unique())

df9.location = df9.locality.apply(lambda x: x.strip())
location_stats = df9['locality'].value_counts(ascending=False)
location_stats

location_stats.values.sum()


len(location_stats[location_stats>10])


len(location_stats)

len(location_stats[location_stats<=10])

location_stats_less_than_10 = location_stats[location_stats<=10]
location_stats_less_than_10

df10 = df9.copy()

df10.locality = df10.locality.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df10.locality.unique())

df_filtered = df10.loc[df9['locality'] == 'other']


df10.shape


df10.head(10)

df10.rent.describe()

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('locality'):
        m = np.mean(subdf.rent)
        st = np.std(subdf.rent)
        reduced_df = subdf[(subdf.rent>(m-st)) & (subdf.rent<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
    df11 = df10.copy()

df11 = remove_pps_outliers(df10)

df11.shape

def plot_scatter_chart(df,locality):
    bhk2 = df[(df.locality==locality) & (df.bhk==2)]
    bhk3 = df[(df.locality==locality) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.property_size,bhk2.rent,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.property_size,bhk3.rent,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(locality)
    plt.legend()
    
plot_scatter_chart(df11,"Rajaji Nagar")

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('locality'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.rent),
                'std': np.std(bhk_df.rent),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.rent<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df12 = remove_bhk_outliers(df11)
df12.shape

plot_scatter_chart(df12,"Rajaji Nagar")

import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df8.rent,rwidth=0.8)
plt.xlabel("rent")
plt.ylabel("Count")


df11.bathroom.unique()

plt.hist(df11.bathroom,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")

dummies = pd.get_dummies(df11.locality)
dummies.head(3)

df11 = pd.concat([df11,dummies.drop('other',axis='columns')],axis='columns')
df11.head()

df12 = df11.drop([ 'locality','lease_type','furnishing', 'parking','facing', 'amenities','water_supply', 'building_type' ], axis='columns')
df12.head(2)

df12.shape

X = df12.drop(['rent'],axis='columns')
X.head(3)

X.shape

y = df12.rent
y.head(3)

len(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)

from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, y, cv=cv)

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'fit_intercept': [True, False],
                'n_jobs': [-1, 1, 2, 3]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,y)


def predict_price(locality,property_size,bathroom,bhk):    
    loc_index = np.where(X.columns==locality)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = property_size
    x[1] = bathroom
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]

predict_price('Bellandur',1400, 2, 2)

import pickle
with open('home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)

import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))
