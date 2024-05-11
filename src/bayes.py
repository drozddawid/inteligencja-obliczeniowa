import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def showTwoAttributesPlot(data, yAttribute='sepalLengthCm',
    xAttribute='sepalWidthCm'):
    # Extract columns for plotting
    yAttributeArray = data[yAttribute]
    xAttributeArray = data[xAttribute]
    classes = data['class']

    # Define colors for different classes
    color_map = {
        'Iris-setosa': 'red',
        'Iris-versicolor': 'green',
        'Iris-virginica': 'blue'
    }

    # Map colors to classes
    colors = [color_map[c] for c in classes]

    # Create scatter plot
    plt.scatter(xAttributeArray, yAttributeArray, c=colors)

    # Add labels and title
    plt.xlabel(xAttribute)
    plt.ylabel(yAttribute)
    plt.title(f'{yAttribute} vs {xAttribute}')

    # Add legend
    for class_name, color in color_map.items():
        plt.scatter([], [], color=color, label=class_name)
    plt.legend()

    # Show the plot
    plt.show()


if __name__ == '__main__':
    data = pd.read_csv('iris/iris.csv')

    attributeNames = ["sepalLengthCm", "sepalWidthCm", "petalLengthCm",
                      "petalWidthCm"]
    showTwoAttributesPlot(data, attributeNames[0], attributeNames[2])

    # Split cluestering data to X and Y
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=125
    )

    # Build a Gaussian Classifier
    model = GaussianNB()

    # Model training
    model.fit(X_train, y_train)

    # Predict Output
    predicted = model.predict([X_test[6]])

    print("Actual Value:", y_test[6])
    print("Predicted Value:", predicted[0])
    y_pred = model.predict(X_test)
    accuray = accuracy_score(y_pred, y_test)
    f1 = f1_score(y_pred, y_test, average="weighted")

    print("Accuracy:", accuray)
    print("F1 Score:", f1)
    # show a plot displaying all classes and predictions


    labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.show()
