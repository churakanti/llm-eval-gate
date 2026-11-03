from evalgate.metrics import compute_metrics, percentile
from evalgate.schema import CaseResult

CATS = ["Technical issue", "Billing inquiry"]
PRIS = ["Low", "Medium", "High", "Critical"]


def r(i, gc, pc, gp, pp, latency=1.0, valid=True):
    return CaseResult(id=i, gold_category=gc, gold_priority=gp, pred_category=pc, pred_priority=pp,
                      valid_output=valid, latency_s=latency, raw_output="")


def test_basic_metrics():
    results = [
        r("1", "Technical issue", "Technical issue", "High", "High"),
        r("2", "Technical issue", "Billing inquiry", "Low", "Medium"),
        r("3", "Billing inquiry", "Billing inquiry", "Critical", "Low"),
        r("4", "Billing inquiry", None, "Medium", None, valid=False),
    ]
    m = compute_metrics(results, CATS, PRIS)
    assert m["category_accuracy"] == 0.5
    assert m["priority_accuracy"] == 0.25
    assert m["valid_output_rate"] == 0.75
    # urgent gold = cases 1 and 3; only case 1 predicted urgent
    assert m["urgent_recall"] == 0.5
    # within one: case1 (0), case2 (1) -> 2 of 4
    assert m["priority_within_one"] == 0.5
    # Technical: P=1/1, R=1/2 -> F1 .667 ; Billing: P=1/2, R=1/2 -> .5
    assert abs(m["category_macro_f1"] - (0.6667 + 0.5) / 2) < 0.001
    assert m["confusion_matrix"]["Billing inquiry"]["INVALID"] == 1


def test_percentile_nearest_rank():
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert percentile([1, 2, 3, 4], 50) == 2
    assert percentile([], 95) == 0.0
