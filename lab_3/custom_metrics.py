import numpy as np

def my_metrics(y_true, y_pred):
    TP, TN, FP, FN = 0, 0, 0, 0

    y_true = y_true.tolist()
    y_pred = y_pred.tolist()

    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        else:
            FP += 1

    print('Confusion matrix')

    print(TN, FP)
    print(FN, TP)

    print(f'Accuracy: {(TP + TN) / (TP + TN + FN + FP)}')

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    print(f'Precision: {precision}')

    print(f'Recall: {recall}')

    print(f'F1: {(2 * precision * recall) / (precision + recall)}')
