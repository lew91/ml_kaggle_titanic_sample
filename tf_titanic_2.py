import numpy as np
import pandas as pd
from collections import Counter
import tensorflow as tf 
from tensorflow.keras import models, layers

#import matplotlib.pyplot as plt 

dftrain_raw = pd.read_csv('~/kaggle/titanic/titanic/train.csv')
dftest_raw = pd.read_csv('~/kaggle/titanic/titanic/test.csv')


# outlier detection
def outliers_detect(df, n, features):
    outlier_indices = []

    for col in features:
        # 1st quartiles(25%)
        q1 = np.percentile(df[col], 25)
        # 3rd quartile(75%)
        q3 = np.percentile(df[col], 75)
        # interquartile range(IQR)
        IQR = q3 - q1

        # outlier step
        outlier_step = 1.5 * IQR

        # Determin a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < q1 - outlier_step) | (df[col] > q3 + outlier_step)].index

        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)

    # select obervations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

    return multiple_outliers


outliers_to_drop = outliers_detect(dftrain_raw, 2, ['Age', 'SibSp','Parch','Fare'])
dftrain_raw = dftrain_raw.drop(outliers_to_drop, axis=0).reset_index(drop=True)


train_len = len(dftrain_raw)
dataset = pd.concat([dftrain_raw, dftest_raw], axis=0, sort=False).reset_index(drop=True)



# Fare
# Fill Fare missing values with the median value
dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
dataset['Fare'] = dataset['Fare'].map(lambda i : np.log(i) if i > 0 else 0)

# Embarked
dataset['Embarked'] = dataset['Embarked'].fillna('S')

# Sex
dataset['Sex'] = dataset['Sex'].map({'male': 0, 'female': 1})

# Age
# Filling missing value of age
# Fill Age with the median age of similar rows according to Pclass, Parch and SibSp.
# index of Nan age rows
index_NaN_age = list(dataset['Age'][dataset['Age'].isnull()].index)

for i in index_NaN_age:
    age_med = dataset['Age'].median()
    age_pred = dataset['Age'][((dataset['SibSp'] == dataset.iloc[i]['SibSp']) &
                               (dataset['Parch'] == dataset.iloc[i]['Parch']) &
                               (dataset['Pclass'] == dataset.iloc[i]['Pclass']))].median()
    if not np.isnan(age_pred):
        dataset['Age'].iloc[i] = age_pred
    else:
        dataset['Age'].iloc[i] = age_med

# Get Title from Name
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset['Title'] = pd.Series(dataset_title)
# Convert to categorical values Title 
dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess',
                                             'Capt', 'Col','Don', 'Dr', 'Major',
                                             'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1,
                                         "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
dataset["Title"] = dataset["Title"].astype(int)

# Drop Name variable
dataset.drop(labels = ["Name"], axis = 1, inplace = True)

# Create a family size descriptor from SibSp and Parch
dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1

# Create new feature of family size
dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)
dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if  s == 2  else 0)
dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)

# convert to indicator values Title and Embarked 
dataset = pd.get_dummies(dataset, columns = ["Title"])
dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em")

# Replace the Cabin number by the type of cabin 'X' if not
dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin']])
dataset = pd.get_dummies(dataset, columns = ["Cabin"],prefix="Cabin")


## Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X. 

Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
    else:
        Ticket.append("X")
        
dataset["Ticket"] = Ticket
dataset["Ticket"].head()

dataset = pd.get_dummies(dataset, columns = ["Ticket"], prefix="T")


# Create categorical values for Pclass
dataset["Pclass"] = dataset["Pclass"].astype("category")
dataset = pd.get_dummies(dataset, columns = ["Pclass"],prefix="Pc")



#####################################
# Drop useless variables 
dataset.drop(labels = ["PassengerId"], axis=1, inplace = True)
test_predict = dftest_raw['PassengerId']

## Separate train dataset and test dataset
train = dataset[:train_len]
test = dataset[train_len:]
test.drop(labels=["Survived"],axis = 1,inplace=True)

## Separate train features and label 
train["Survived"] = train["Survived"].astype(int)
Y_train = train["Survived"]
X_train = train.drop(labels = ["Survived"],axis = 1)

tf.keras.backend.clear_session()

model = models.Sequential()
model.add(layers.Dense(100, activation='relu', input_shape=(66, )))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',
              loss = 'binary_crossentropy',
              metrics=['accuracy','AUC'])

history = model.fit(X_train, Y_train, batch_size=64, epochs=100,
                    validation_split = 0.2)

# def plot_metric(history, metric):
#     train_metrics = history.history[metric]
#     val_metrics = history.history['val_'+metric]
#     epochs = range(1, len(train_metrics) + 1)
#     plt.plot(epochs, train_metrics, 'bo--')
#     plt.plot(epochs, val_metrics, 'ro-')
#     plt.title('Training and validation '+ metric)
#     plt.xlabel("Epochs")
#     plt.ylabel(metric)
#     plt.legend(["train_"+metric, 'val_'+metric])
#     plt.show()


print(model.predict_classes(test[0:10]))
#print(test_predict[0:10])

y_predict = model.predict_classes(test)
result_predicted = pd.DataFrame(y_predict)
print(result_predicted[:5])
result = pd.concat([test_predict, result_predicted], axis=1)
result.columns =['PassengerId','Survived']
print(result[:5])
result.to_csv('Submission.csv',index=False)


#plot_metric(history, 'loss')
#plt.ion()
#plot_metric(history, 'AUC')















