import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)
x = np.random.rand(100).reshape(-1, 1)
y_train = [1 if v <= 0.5 else 2 for v in x[:50]]
y_test = [1 if v <= 0.5 else 2 for v in x[50:]]

for k in [1, 2, 3, 4, 5, 20, 30]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x[:50], y_train)
    preds = model.predict(x[50:])
    acc = accuracy_score(y_test, preds)
    print(f"k={k}, Accuracy={acc:.2f}")

    # Plot for each k
    plt.figure(figsize=(6,4))
    plt.scatter(x[:50], y_train, color='blue', label='Train')
    plt.scatter(x[50:], preds, color='red', marker='x', label='Predicted')
    plt.title(f"k-NN Classification (k={k})")
    plt.xlabel("x values")
    plt.ylabel("Class")
    plt.legend()
    plt.grid()
    plt.show()