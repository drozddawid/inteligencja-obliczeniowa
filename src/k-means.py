import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pygal
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix


def showTwoAttributesPlot(data, yAttribute='sepalLengthCm', xAttribute='sepalWidthCm'):
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


def showElbowMethodPlot(data_to_cluster):
    intertias = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i)
        km.fit(data_to_cluster)
        # inertia - suma kwadratów odległości od każdego punktu do przypisanego mu centroidu
        intertias.append(km.inertia_)

    plt.plot(range(1, 11), intertias)
    plt.xlabel("K")
    plt.ylabel("Wartość bezwładności")
    plt.title("Metoda łokciowa")
    plt.show()


if __name__ == '__main__':
    data = pd.read_csv('iris/iris.csv')

    attributeNames = ["sepalLengthCm", "sepalWidthCm", "petalLengthCm", "petalWidthCm"]
    # showTwoAttributesPlot(data, attributeNames[0], attributeNames[2])

    clustering_data = data.iloc[:, [0, 1, 2, 3]]
    showElbowMethodPlot(clustering_data)

    k_means = KMeans(n_clusters=3, init='k-means++', random_state=12).fit(clustering_data)
    data_with_centroids = data.copy()
    for cc in k_means.cluster_centers_:
        row = pd.DataFrame([{"sepalLengthCm": cc[0], "sepalWidthCm": cc[1], "petalLengthCm": cc[2], "petalWidthCm": cc[3], 'class': "centroids"}])
        data_with_centroids = pd.concat([data_with_centroids, row], ignore_index=True)

    sns.set_style("whitegrid")
    sns.pairplot(data_with_centroids, hue="class", height=3)
    plt.show()

    data_with_prediction = data.copy()
    prediction = k_means.predict(clustering_data)
    data_with_prediction['class'] = prediction
    data['prediction'] = prediction

    for cc in k_means.cluster_centers_:
        row = pd.DataFrame([{"sepalLengthCm": cc[0], "sepalWidthCm": cc[1], "petalLengthCm": cc[2], "petalWidthCm": cc[3], 'class': "centroids"}])
        data_with_prediction = pd.concat([data_with_prediction, row], ignore_index=True)

    sns.set_style("whitegrid")
    sns.pairplot(data_with_prediction, hue="class", height=3)
    plt.show()

    class_to_specie = {
        0: 'Iris-setosa',
        1: 'Iris-virginica',
        2: 'Iris-versicolor'
    }

    actual = data['class'].values
    prediction = [class_to_specie[i] for i in data['prediction'].values]

    predicted_good = 0
    for act, pred in zip(actual, prediction):
        print(f"{act} - {pred}")
        if act == pred:
            predicted_good += 1

    print("Predicted good: ", predicted_good / len(actual))

    cm = confusion_matrix(actual, prediction)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)

    class_names = ['Iris-setosa', 'Iris-virginica', 'Iris-versicolor']
    tick_marks = np.arange(len(class_names)) + 0.5
    plt.xticks(tick_marks, class_names, rotation=0)
    plt.yticks(tick_marks, class_names, rotation=0)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    clusters = clustering_data.copy()
