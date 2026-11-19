
import json

with open("results/face_crop_comparison.json", "r") as file:
    data = json.load(file)

results = data["results"]
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def evaluate_threshold(results, score_type, threshold):
    tp = 0  # Fake correctly predicted as fake
    fn = 0  # Fake predicted as real
    fp = 0  # Real predicted as fake
    tn = 0  # Real correctly predicted as real

    for item in results:
        actual = item["true_label"]
        fake_score = item[score_type][0]
        predicted = "fake" if fake_score >= threshold else "real"

        if actual == "fake" and predicted == "fake":
            tp += 1
        elif actual == "fake" and predicted == "real":
            fn += 1
        elif actual == "real" and predicted == "fake":
            fp += 1
        else:
            tn += 1

    total = tp + fn + fp + tn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )
    specificity = tn / (tn + fp) if tn + fp else 0

    return accuracy, precision, recall, f1, specificity, tp, fn, fp, tn


for score_type, title in [
    ("face", "FACE-CROP"),
    ("full", "FULL-FRAME")
]:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print(
        "Threshold | Accuracy | Fake Precision | Fake Recall | "
        "Fake F1 | Real Specificity | TP FN FP TN"
    )

    for threshold in thresholds:
        metrics = evaluate_threshold(results, score_type, threshold)
        accuracy, precision, recall, f1, specificity, tp, fn, fp, tn = metrics

        print(
            f"{threshold:9.1f} | "
            f"{accuracy:8.2%} | "
            f"{precision:14.2%} | "
            f"{recall:11.2%} | "
            f"{f1:7.2%} | "
            f"{specificity:15.2%} | "
            f"{tp:2} {fn:2} {fp:2} {tn:2}"
        )
