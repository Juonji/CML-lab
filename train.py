import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from xgboost import XGBClassifier


df = pd.read_csv("titanic.csv")
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Age'] = df['Age'].fillna(df['Age'].median())


df['family_size'] = df['SibSp'] + df['Parch'] + 1
df['is_child'] = (df['Age'] < 16).astype(int)


df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'family_size', 'is_child']]


X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=67
)


model = XGBClassifier()
model.fit(X_train, y_train)


pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print("Accuracy:", acc)


cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("plot.png")
