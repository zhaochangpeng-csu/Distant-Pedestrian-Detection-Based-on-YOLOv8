import numpy as np
from pedestrian.metrics import overlap, match_image, evaluate_records


def record(pred=(), boxes=(), ignores=()):
    return {"pred":pred,"boxes":boxes,"ignores":ignores}


def test_perfect_and_duplicate():
    box=[0,0,20,20]
    m=evaluate_records([record([[*box,.9]],[box])])
    assert m["map50_95"]==1 and m["ap_small"]==1
    m=evaluate_records([record([[*box,.9],[*box,.8]],[box])])
    assert m["tp"]==1 and m["fp"]==1 and m["precision"]==.5


def test_ignore_is_ioa_not_iou():
    m=evaluate_records([record([[10,10,20,20,.9]],[],[[0,0,100,100]])])
    assert m["fp"]==0 and m["ignored_predictions_at_iou50"]==1
    assert m["map50_95"] is None


def test_positive_priority_over_ignore():
    b=[0,0,10,10]
    scores,tp,fp,n,ignored=match_image([[*b,.9]],[b],[b])
    assert tp.tolist()==[1] and n==1 and ignored==0


def test_matching_is_per_iou_threshold():
    _,tp,fp,n,ignored=match_image([[0,0,10,8,.9]],[[0,0,10,10]],[],.85)
    assert fp.tolist()==[1]
    assert evaluate_records([record([[0,0,10,8,.9]],[[0,0,10,10]])])["map50_95"]<1


def test_large_gt_ignored_in_small_metric():
    r=record([[0,0,100,100,.99],[110,0,120,10,.8]],[[0,0,100,100],[110,0,120,10]])
    assert evaluate_records([r])["ap_small"]==1


def test_empty_predictions_and_negative_image():
    m=evaluate_records([record([],[[0,0,10,10]]),record([[0,0,10,10,.9]],[])])
    assert m["map50_95"]==0 and m["fn"]==1 and m["fp"]==1
    assert evaluate_records([])["map50"] is None


def test_score_order_independent_of_input_order():
    r=record([[40,40,50,50,.1],[0,0,10,10,.9]],[[0,0,10,10]])
    assert evaluate_records([r])["map50"]==1


def test_overlap_empty_shapes():
    assert overlap([],[]).shape==(0,0)
    assert overlap([[0,0,1,1]],[]).shape==(1,0)
