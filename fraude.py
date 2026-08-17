import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("creditcard.csv")

print(data.shape)
print(data["Class"].value_counts())

# tres peu de fraudes par rapport au total, faut faire attention avec ca
X = data.drop(columns=["Class"])
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modele = RandomForestClassifier(n_estimators=100, random_state=42)
modele.fit(X_train, y_train)

predictions = modele.predict(X_test)
precision = accuracy_score(y_test, predictions)
print(f"Accuracy : {precision * 100:.2f}%")

# accuracy seule veut rien dire ici (trop peu de fraudes), du coup je regarde aussi precision et recall
print(classification_report(y_test, predictions))