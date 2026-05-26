import json

from oraclematch.experiments.run_mvp import main


def test_version_command(capsys):
    assert main(["version"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert "oraclematch" in out


def test_demo_runs_and_is_mock(capsys):
    assert main(["demo", "--generations", "4", "--population", "8"]) == 0
    out = json.loads(capsys.readouterr().out)
    demo = out["results"][0]
    assert demo["backend"] == "mock"
    assert demo["best_fitness"] is not None
    assert demo["n_evaluations"] > 0


def test_demo_ablation(capsys):
    assert main(["demo", "--ablation", "--generations", "4", "--population", "8"]) == 0
    out = json.loads(capsys.readouterr().out)
    labels = [r["experiment"] for r in out["results"]]
    assert "ablation_qd_vs_ga" in labels
    abl = next(r for r in out["results"] if r["experiment"] == "ablation_qd_vs_ga")
    assert abl["qd"]["coverage_cells"] >= abl["ga"]["coverage_cells"]


def test_default_command_is_demo(capsys):
    assert main([]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["results"][0]["experiment"] == "demo_map_elites"
