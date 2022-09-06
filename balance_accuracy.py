def balance_accuracy(gt_ls, pred_ls):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(gt_ls)):
        if gt_ls[i] == 1 and pred_ls[i] == 1:
            tp += 1
        elif gt_ls[i] == 1 and pred_ls[i] == 0:
            fn += 1
        elif gt_ls[i] == 0 and pred_ls[i] == 1:
            fp += 1
        else:
            tn += 1
    balanced_accuracy = (tp/(tp + fn) + tn/(tn + fp))/2
    return balanced_accuracy
