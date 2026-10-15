import pytest
from pedestrian.training import PersonValidator
from pedestrian.metrics import evaluate_records,threshold_sweep


def test_best_checkpoint_uses_map_not_upstream_weighted_fitness(tmp_path):
    validator=PersonValidator(save_dir=tmp_path,args={"plots":False})
    validator.records=[{"pred":[[0,0,10,6,.9]],"boxes":[[0,0,10,10]],"ignores":[]}]
    validator.metrics.names={0:"person"}
    result=validator.get_stats()
    expected=evaluate_records(validator.records)
    assert result["fitness"]==pytest.approx(expected["map50_95"])
    assert result["fitness"]!=pytest.approx(validator.metrics.fitness)


def test_threshold_sweep_same_working_points():
    records=[{"pred":[[0,0,10,10,.9],[20,20,30,30,.2]],"boxes":[[0,0,10,10]],"ignores":[]}]
    for row in threshold_sweep(records,[.1,.25,.95]):
        expected=evaluate_records(records,row["operating_confidence"])
        for key in ("precision","recall","f1","tp","fp","fn"):
            assert row[key]==pytest.approx(expected[key])
