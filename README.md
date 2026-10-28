# FUTURE_ML_02_Support_Ticket_Classification

# FUTURE_ML_02 – Support Ticket Classification

A Natural Language Processing (NLP) project that classifies customer support tickets into categories using machine learning.

**Future Interns – Machine Learning Internship | Task 02**

## Project Overview

Customer support teams receive tickets about billing, cancellations, refunds, product questions, and technical problems. Sorting them by hand is slow. This project reads the text of a ticket and automatically predicts its category using text preprocessing, TF-IDF, and a classification model.

## Ticket Categories

- Billing Inquiry
- Cancellation Request
- Product Inquiry
- Refund Request
- Technical Issue

## Dataset

- **File:** `[your_dataset_name].csv`
- **Total tickets:** [number]
- **Input:** ticket text
- **Target:** ticket category (5 classes)

## Tools Used

| Tool | Purpose |
|---|---|
| Python | Programming language |
| Pandas, NumPy | Data handling |
| Scikit-learn | TF-IDF, model training, evaluation |
| Matplotlib, Seaborn | Charts and confusion matrix |
| Jupyter Notebook / Google Colab | Development |

## Project Workflow

1. **Data exploration:** checked the data structure, missing values, and category counts.
2. **Text preprocessing:** converted text to lowercase and removed symbols and extra spaces.
3. **Feature extraction:** converted text into numbers using TF-IDF.
4. **Model training:** trained [model name, e.g. Logistic Regression] on the training data.
5. **Evaluation:** tested on unseen tickets using accuracy, precision, recall, F1-score, and a confusion matrix.

## Results

| Metric | Score |
|---|---|
| Accuracy | [XX%] |
| Precision (weighted) | [XX%] |
| Recall (weighted) | [XX%] |
| F1-score (weighted) | [XX%] |

**Observations**
- Best performing category: [category name]
- Most often confused categories: [category A and category B]

### Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

## Repository Structure

```text
FUTURE_ML_02_Support_Ticket_Classification/
├── README.md
├── Support_Ticket_Classification.ipynb
├── [your_dataset_name].csv
└── images/
    └── confusion_matrix.png
```

## How to Run

1. Download or clone this repository.
2. Install the libraries: `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Open `Support_Ticket_Classification.ipynb` in Jupyter Notebook or Google Colab.
4. Run all cells from top to bottom.

## Key Learnings

- Cleaning and preparing text data for machine learning
- Using TF-IDF to turn text into features
- Training and evaluating a text classification model
- Reading a confusion matrix to find common mistakes

## Future Improvements

- Try other models such as SVM or Naive Bayes
- Tune hyperparameters to improve accuracy
- Handle class imbalance if present
- Test on new, real-world tickets

## Author

**Tanisha**
BCA Data Science Student
GitHub: [Tanisha82-bit](https://github.com/Tanisha82-bit)

## Internship

Future Interns – Machine Learning Internship, Task 02
