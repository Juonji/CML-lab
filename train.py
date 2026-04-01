import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from xgboost import XGBClassifier


df = pd.read_csv("titanic.csv")
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch']]
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Age'] = df['Age'].fillna(df['Age'].median())


df['family_size'] = df['SibSp'] + df['Parch'] + 1
df['is_child'] = (df['Age'] < 16).astype(int)


df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'family_size', 'is_child']]


df['Embarked'] = df['Embarked'].fillna('S')


df = pd.get_dummies(df, columns=['Embarked'])


df['age_bin'] = pd.cut(
    df['Age'],
    bins=[0, 12, 18, 35, 60, 100],
    labels=False
)


X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=67
)


params = {
    'max_depth':[3,4,5],
    'learning_rate':[0.01,0.05,0.1],
    'n_estimators':[100,200],
    'subsample':[0.8,1.0]
}


model = XGBClassifier(n_estimators=50, max_depth=2)


grid = GridSearchCV(
    model,
    params,
    cv=5,
    scoring="accuracy"
)


grid.fit(X_train, y_train)
best_model = grid.best_estimator_
pred = best_model.predict(X_test)
acc = accuracy_score(y_test, pred)


print("Best params:", grid.best_params_)
print("Accuracy:", acc)



cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("plot.png")
