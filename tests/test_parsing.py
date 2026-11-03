from evalgate.triage import extract_json_object, parse_triage_output

CATS = ["Technical issue", "Billing inquiry", "Product inquiry", "Refund request", "Cancellation request"]
PRIS = ["Low", "Medium", "High", "Critical"]


def test_plain_json():
    res, err = parse_triage_output('{"category": "Billing inquiry", "priority": "High"}', CATS, PRIS)
    assert err is None and res.category == "Billing inquiry" and res.priority == "High"


def test_code_fence_and_chatter():
    raw = 'Sure! Here is the result:\n```json\n{"category": "Refund request", "priority": "Low"}\n```\nHope that helps.'
    res, err = parse_triage_output(raw, CATS, PRIS)
    assert err is None and res.category == "Refund request"


def test_case_and_synonym_normalization():
    res, err = parse_triage_output('{"Category": "billing", "Priority": "URGENT"}', CATS, PRIS)
    assert err is None and res.category == "Billing inquiry" and res.priority == "Critical"


def test_label_outside_allowed_set_is_invalid():
    res, err = parse_triage_output('{"category": "Shipping", "priority": "High"}', CATS, PRIS)
    assert res is None and "not in allowed set" in err


def test_non_json_is_invalid():
    res, err = parse_triage_output("I think this is a billing problem.", CATS, PRIS)
    assert res is None and "no JSON" in err


def test_extract_ignores_broken_then_finds_valid_object():
    assert extract_json_object('{broken {"category": "x"} tail') == {"category": "x"}
