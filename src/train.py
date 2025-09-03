# src/train.py
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import datasets, tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def main(test_size=0.2, random_state=42, save_outputs=True, print_metrics=False):
    """Train a Decision Tree on Iris dataset, optionally save plots, optionally print metrics"""
    # Load Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train Decision Tree
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate accuracy
    acc = accuracy_score(y_test, y_pred)

    if print_metrics:
        # Print confusion matrix & classification report
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:\n", cm)
        report = classification_report(y_test, y_pred, target_names=iris.target_names)
        print("Classification Report:\n", report)

        # Feature importance
        feature_importances = model.feature_importances_
        features = iris.feature_names
        importance_table = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
        importance_table = importance_table.sort_values(by='Importance', ascending=False)
        print("\nFeature Importance:\n", importance_table)

    if save_outputs:
        # Create outputs folder
        os.makedirs("outputs", exist_ok=True)

        # Confusion matrix heatmap
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=iris.target_names,
                    yticklabels=iris.target_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig("outputs/confusion_matrix.png")
        plt.close()

        # Decision tree visualization
        plt.figure(figsize=(12, 8))
        tree.plot_tree(
            model,
            feature_names=iris.feature_names,
            class_names=iris.target_names,
            filled=True
        )
        plt.title("Decision Tree: Iris Classifier")
        plt.savefig("outputs/decision_tree.png")
        plt.close()

        # Petal scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(iris.data[:, 2], iris.data[:, 3], c=iris.target, cmap="viridis", edgecolor="k")
        plt.xlabel("Petal length (cm)")
        plt.ylabel("Petal width (cm)")
        plt.title("Iris dataset: Petal length vs Petal width")
        plt.colorbar(label="Species")
        plt.savefig("outputs/petal_scatter.png")
        plt.close()

        # Pair plot
        iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
        iris_df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
        sns.pairplot(iris_df, hue="species", palette="bright")
        plt.savefig("outputs/pairplot.png")
        plt.close()

        # Feature histograms
        plt.figure(figsize=(12, 8))
        for i, feature in enumerate(iris.feature_names, 1):
            plt.subplot(2, 2, i)
            sns.histplot(
                data=iris_df,
                x=feature,
                hue="species",
                multiple="stack",
                palette="Set2",
                edgecolor="black"
            )
            plt.title(f"{feature} distribution by species")
        plt.tight_layout()
        plt.savefig("outputs/feature_histograms.png")
        plt.close()

        # Save trained model
        joblib.dump(model, "outputs/iris_model.joblib")

        if print_metrics:
            print("\nAll plots and model saved to 'outputs/' folder.")

    return acc


# CLI: Only run if executed directly
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a Decision Tree on Iris dataset")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    accuracy = main(test_size=args.test_size,
                    random_state=args.random_state,
                    save_outputs=True,
                    print_metrics=True)
    print("Accuracy:", accuracy)