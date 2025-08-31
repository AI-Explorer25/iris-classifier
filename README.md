# Iris Classifier (Decision Tree)

## Overview
This project is an end-to-end machine learning example that builds a Decision Tree classifier for the classic Iris dataset using Python and scikit-learn.  

It includes:

- Loading and splitting the Iris dataset into training and testing sets
- Training a Decision Tree classifier
- Evaluating model performance (accuracy, confusion matrix, classification report)
- Visualising results:
  - Confusion matrix heatmap
  - Decision tree plot
  - Petal length vs width scatter plot
  - Pair plot of all features
  - Feature histograms
- Feature importance ranking

This project demonstrates a complete ML workflow in Python with reproducible results via a CLI script.

## Quick start
```bash
# Clone the repository
git clone https://github.com/<YOUR_USERNAME>/iris-classifier.git
cd iris-classifier

# Create a virtual environment
python -m venv venv

# Activate the virtual environment

# Windows (Anaconda Prompt or cmd.exe)
.\venv\Scripts\activate

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# macOS/Linux
source venv/bin/activate

# Upgrade pip to ensure latest versions
python -m pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# Run the training script
python src/train.py --test-size 0.2 --random-state 42

# Run Tests

set PYTHONPATH=.
pytest tests/test_train.py -v
```
To verify the training script works correctly, a pytest is included. Make sure to set the Python path so the src module can be found, then run the test.
The test ensures that the Decision Tree achieves an accuracy of at least 0.9 on the Iris dataset, confirming the model works as expected. If the test passes, you should see an output like:

tests/test_train.py::test_accuracy PASSED

## License
Distributed under the MIT License. See [LICENSE](LICENSE) for more information.