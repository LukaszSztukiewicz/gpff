import json
import random
import numpy as np
import pytest
import torch
from torch import nn
from schnetpack import properties as P
from gpff import keys, vocab
from gpff.sample.base import Calculator
from gpff.sample.absorbing import AbsorbingDirectDenoisingSampler
from gpff.sample.absorbing_study import CoupledAbsorbingSampler, ScheduledAbsorbingSampler
from gpff.sample.remdm import transition_probabilities
from gpff.sample.formula import FormulaPrior, build_formula_prior
from gpff.analysis.absorbing_diagnostics import DecisionDiagnostics, snapshot, probe
from gpff.model.recovery import RecoveryAugmentation
from gpff.model.initialization import initialize_weights
from gpff.model.painn import PaiNN
from gpff.pseudo_pes.weights import SigmaWeights
from gpff.analysis.type_conditionals import fixed_geometry_decode


def batch(n=12, size=5):
    return {P.R: torch.randn(n*size, 3), P.Z: torch.full((n*size,), vocab.MASK_Z),
            P.idx_m: torch.arange(n).repeat_interleave(size), P.n_atoms: torch.full((n,), size),
            keys.nfe: torch.zeros(n, dtype=torch.int32), keys.converged: torch.zeros(n, dtype=torch.bool),
            P.idx_i: torch.tensor([], dtype=torch.long), P.idx_j: torch.tensor([], dtype=torch.long)}


class Model(nn.Module):
    def __init__(self, stochastic=False):
        super().__init__()
        self.stochastic = stochastic
        self.calls = []

    def forward(self, b):
        self.calls.append({k: v.clone() for k, v in b.items() if k in (P.Z, P.R, keys.sigma_curr)})
        if self.stochastic:
            torch.rand(3)
            random.random()
            np.random.rand()
        p = torch.zeros(len(b[P.Z]), vocab.n_tokens(), device=b[P.R].device)
        p[:, vocab.TOKEN_Z.index(7)] = 1
        return {keys.forces: b[P.Z][:,None].expand(-1,3).to(b[P.R].dtype) / 100,
                "type_probabilities": p}


@pytest.mark.parametrize("r,s,strength", [(1,.8,1),(.8,.5,1),(.5,.4,.7),(.1,0,1),(0,0,1),(1,0,0),(1,0,1)])
def test_remdm_identity_and_endpoints(r,s,strength):
    a,b = transition_probabilities(r,s,strength)
    assert 0 <= a <= 1 and 0 <= b <= 1
    assert r*(1-a) + (1-r)*b == pytest.approx(s)


@pytest.mark.parametrize("args", [(.2,.4,.5),(.5,.1,1.1),(float("nan"),0,0),(-1,0,0)])
def test_invalid_transitions_refused(args):
    with pytest.raises(ValueError):
        transition_probabilities(*args)


def test_mask_marginals_statistically_and_no_fresh_remasking():
    sampler = CoupledAbsorbingSampler(n_steps=8, type_schedule="linear", remask_kernel="random", remask_prob=1)
    work = batch(n=20000, size=1)
    sampler._init_state(work)
    p = Model()(work)["type_probabilities"]
    for step in range(8,0,-1):
        before = work[P.Z].clone()
        sampler._transition(work,p,step)
        assert not bool((sampler.last_selected & sampler.last_remasked).any())
        assert bool((before[sampler.last_remasked] != vocab.MASK_Z).all())
        assert float((work[P.Z] == vocab.MASK_Z).float().mean()) == pytest.approx(sampler.masking_rate(step), abs=.015)
    assert not bool((work[P.Z] == vocab.MASK_Z).any())


def test_remasking_runs_without_a_new_decode():
    sampler = CoupledAbsorbingSampler(n_steps=8, remask_kernel="random", remask_prob=1)
    work = batch(n=1000, size=1)
    work[P.Z].fill_(6)
    sampler._init_state(work)
    sampler._transition(work, Model()(work)["type_probabilities"], 4)
    assert not bool(sampler.last_selected.any())
    assert bool(sampler.last_remasked.any())


def test_confidence_uses_saved_decoding_probability():
    work = batch(n=2000, size=1)
    work[P.Z].fill_(6)
    sampler = CoupledAbsorbingSampler(n_steps=8, remask_kernel="saved_confidence", remask_prob=1)
    sampler._init_state(work)
    sampler._decode_confidence[::2] = 0
    sampler._transition(work, Model()(work)["type_probabilities"], 4)
    assert bool((work[P.Z][1::2] == 6).all())
    assert bool((work[P.Z][::2] == vocab.MASK_Z).any())


def test_t1_uses_updated_types_at_identical_coordinates_counts_nfe():
    model = Model()
    calc = Calculator(model=model, device="cpu")
    sampler = CoupledAbsorbingSampler(calculator=calc, n_steps=1, recompute_forces=True,
                                      remask_kernel="random", stochastic_lambda=0)
    out = sampler.denoise(batch(2,3))
    assert len(model.calls) == 2
    torch.testing.assert_close(model.calls[0][P.R],model.calls[1][P.R])
    assert bool((model.calls[0][P.Z] == vocab.MASK_Z).all())
    assert bool((model.calls[1][P.Z] == 7).all())
    assert out[keys.nfe].tolist() == [2,2]
    torch.testing.assert_close(out[P.R], model.calls[0][P.R] + .035)


def test_float64_prior_confidence_state_matches_float32_calculator():
    state = batch(2,3)
    state[P.R] = state[P.R].double()
    sampler = CoupledAbsorbingSampler(calculator=Calculator(Model(),device="cpu"),
        n_steps=2, remask_kernel="saved_confidence", remask_prob=.2, recompute_forces=True)
    out = sampler.denoise(state)
    assert sampler._decode_confidence.dtype == torch.float32
    assert not bool((out[P.Z] == vocab.MASK_Z).any())


def test_nonfinite_force_estimates_do_not_poison_diagnostic_json(tmp_path):
    state = batch(2,3)
    calc = Calculator(Model(),device="cpu")
    sampler = CoupledAbsorbingSampler(calculator=calc,n_steps=2)
    diag = DecisionDiagnostics(tmp_path,steps=[2],compare_context=False)
    pred = calc.calculate(state,forces=True,x0=True,estimate_sigma=True)
    pred[keys.sigma_curr][:3] = float("nan")
    diag.record(sampler,state,pred,2,"decision_input")
    row=json.loads((tmp_path/"decisions.jsonl").read_text())
    assert row["excluded_molecules"] == 1
    assert row["finite_molecules"] == 1
    assert row["slots"] == 3


def test_diagnostics_preserve_rng_state_and_capture_decision_input(tmp_path):
    base = batch(2,3)
    base[P.Z][0] = 6
    def run(diagnostics):
        torch.manual_seed(123)
        random.seed(123)
        np.random.seed(123)
        calc = Calculator(model=Model(stochastic=True),device="cpu")
        sampler = CoupledAbsorbingSampler(calculator=calc, n_steps=2, diagnostics=diagnostics)
        out = sampler.denoise(snapshot(base))
        return out, torch.get_rng_state(), random.getstate(), np.random.get_state(), sampler._generator(torch.device("cpu")).get_state(),calc.model.calls
    plain = run(None)
    diag = DecisionDiagnostics(tmp_path,steps=[2,1],max_states=4)
    instrumented = run(diag)
    torch.testing.assert_close(plain[0][P.R],instrumented[0][P.R])
    torch.testing.assert_close(plain[0][P.Z],instrumented[0][P.Z])
    for i in (1,4):
        assert torch.equal(plain[i],instrumented[i])
    assert plain[2] == instrumented[2]
    np.testing.assert_equal(plain[3],instrumented[3])
    state = torch.load(tmp_path / "state_00000.pt",weights_only=False)
    torch.testing.assert_close(state["positions"],instrumented[5][0][P.R])
    assert not bool(state["context_slots"][0])
    rows = [json.loads(s) for s in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    first = rows[0]
    assert first["context_slots"] == 5
    assert sum(first["partial_context_mass"]) == pytest.approx(sum(first["remasked_context_mass"]))


def test_legacy_study_matches_original_without_diagnostics():
    state = batch(3,4)
    outputs = []
    for cls in (AbsorbingDirectDenoisingSampler,CoupledAbsorbingSampler):
        torch.manual_seed(7)
        sampler = cls(calculator=Calculator(Model(),device="cpu"),n_steps=4,type_seed=7)
        outputs.append(sampler.denoise(snapshot(state)))
    for k in (P.R,P.Z,keys.nfe):
        torch.testing.assert_close(outputs[0][k],outputs[1][k])


@pytest.mark.parametrize("mode",["rollout","isotropic"])
def test_t2_reconstructs_clean_target_keeps_initial_sigma_context_weights(mode):
    work = batch(2,4)
    work[keys.x_0] = work[P.R] + .5
    work[keys.forces] = 2*(work[keys.x_0]-work[P.R])
    work[keys.energy] = torch.ones(2)
    work[keys.sigma_curr] = torch.tensor([.5]*4+[2.]*4)[:,None]
    work[keys.type_target] = torch.zeros(8,vocab.n_tokens())
    teacher = Model()
    aug = RecoveryAugmentation(mode=mode,probability=1,weight_generator=SigmaWeights(max_clipping=1))
    out,metrics = aug(work,teacher,0,0)
    torch.testing.assert_close(out[P.R]+out[keys.forces]/2,work[keys.x_0])
    torch.testing.assert_close(out[keys.sigma_curr],work[keys.sigma_curr])
    torch.testing.assert_close(out[P.Z],work[P.Z])
    torch.testing.assert_close(out[keys.type_target],work[keys.type_target])
    torch.testing.assert_close(out[keys.forces_weight],torch.tensor([1.]*4+[.25]*4)[:,None])


def test_t2_isotropic_matches_each_teacher_displacement_norm_without_touching_global_rng():
    class StructuredTeacher(nn.Module):
        def forward(self,inputs):
            return {keys.forces:-.5*inputs[P.R]}
    state=batch(4,7)
    state[keys.x_0]=state[P.R]+.3
    state[keys.energy]=torch.ones(4)
    state[keys.sigma_curr]=torch.arange(1,5.).repeat_interleave(7)[:,None]
    teacher=StructuredTeacher()
    rng_state=torch.get_rng_state()
    rollout,_=RecoveryAugmentation(mode="rollout",probability=1)(state,teacher,2,3)
    isotropic,_=RecoveryAugmentation(mode="isotropic",probability=1)(state,teacher,2,3)
    assert torch.equal(torch.get_rng_state(),rng_state)
    a=rollout[P.R]-state[P.R]
    b=isotropic[P.R]-state[P.R]
    assert float(a.norm())>0
    assert not torch.allclose(a,b)
    for m in range(4):
        slots=state[P.idx_m]==m
        torch.testing.assert_close(a[slots].norm(),b[slots].norm())
        torch.testing.assert_close(a[slots].mean(0),torch.zeros(3),atol=1e-7,rtol=0)
        torch.testing.assert_close(b[slots].mean(0),torch.zeros(3),atol=1e-7,rtol=0)
    torch.testing.assert_close(rollout[P.R]+rollout[keys.forces]/2,state[keys.x_0])
    torch.testing.assert_close(isotropic[P.R]+isotropic[keys.forces]/2,state[keys.x_0])


def test_sigma_is_declared_schedule_not_force_estimate():
    model = Model()
    sampler = ScheduledAbsorbingSampler(calculator=Calculator(model,device="cpu"), n_steps=3)
    sampler.denoise(batch(2,3))
    actual = [float(c[keys.sigma_curr][0]) for c in model.calls]
    assert actual == pytest.approx([sampler.noise_schedule.get_sigma(s) for s in [3,2,1]])


def test_sigma_embedding_broadcast_floor_and_disabled_compatibility():
    from schnetpack.nn import GaussianRBF,CosineCutoff
    def model(enabled):
        return PaiNN(8,1,GaussianRBF(4,5),CosineCutoff(5),sigma_conditioning=enabled)
    old, new = model(False), model(True)
    assert not any("sigma_embedding" in k for k in old.state_dict())
    observed=[]
    new.sigma_embedding.register_forward_pre_hook(lambda m,a: observed.append(a[0].clone()))
    inputs = batch(2,2)
    inputs[P.Rij] = torch.zeros(0,3)
    inputs[keys.sigma_curr] = torch.tensor([[0.],[2.]])
    new(inputs)
    torch.testing.assert_close(observed[0].flatten(),torch.tensor([1e-4,1e-4,2.,2.]).log())


def test_formula_training_only_fresh_exchangeable_and_joint_size(tmp_path):
    from ase import Atoms
    from ase.db import connect
    dataset, split, prior_path = [tmp_path / name for name in ("molecules.db","split.npz","prior.json")]
    with connect(str(dataset)) as db:
        for zs in ([1,6],[1,6],[7,8,1],[9]*4,[8]*5):
            db.write(Atoms(numbers=zs,positions=np.zeros((len(zs),3))))
    np.savez(split,train_idx=[0,1,2],val_idx=[3],test_idx=[4])
    payload = build_formula_prior(dataset,split,prior_path)
    assert payload["training_samples"] == 3 and sum(payload["frequencies"]) == 3
    prior = FormulaPrior(prior_path,mode="joint",seed=1)
    source = batch(100,4)
    source[P.Z].fill_(9)
    source["secret"] = torch.tensor([42])
    out = prior.sample_from_batch(source)
    assert "secret" not in out and not bool((out[P.Z] == 9).any())
    assert set(out[P.n_atoms].tolist()) == {2,3}
    prior.verify(out)
    # A formula batch must be a complete sampler input, including synthetic IDs
    # needed by the real neighbor builder, without borrowing test identities.
    calc=Calculator(Model(),device="cpu")
    neighbors=calc.update_neighbors(calc.cast_batch(out))
    assert P.idx_i in neighbors and P.idx_j in neighbors
    assert out[P.idx].tolist() == list(range(100))
    assert not torch.equal(out[P.R],torch.zeros_like(out[P.R]))
    with pytest.raises(ValueError,match="No training formula"):
        FormulaPrior(prior_path).sample_from_batch(source)
    out[P.Z][0] = 9
    with pytest.raises(ValueError,match="preserve"):
        prior.verify(out)


def test_fixed_geometry_decode_keeps_positions_and_uses_clean_vs_generated_context():
    work = batch(2,3)
    truth = torch.full_like(work[P.Z],6)
    positions = work[P.R].clone()
    model = Model()
    calc=Calculator(model,device="cpu")
    for mode in ("clean","generated"):
        model.calls.clear()
        output,calls = fixed_geometry_decode(calc,work,truth,torch.Generator().manual_seed(4),mode=mode)
        assert calls == 3 and bool((output == 7).all())
        torch.testing.assert_close(model.calls[-1][P.R],positions)
        context = 6 if mode == "clean" else 7
        assert int((model.calls[-1][P.Z] == context).sum()) == 4


@pytest.mark.parametrize("experiment",["gpff_absorbing_recovery","gpff_absorbing_sigma"])
def test_training_config_instantiation_and_strict_initialization(tmp_path,experiment):
    import hydra
    from hydra import compose,initialize_config_dir
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]/"src/gpff/configs"
    with initialize_config_dir(config_dir=str(root),version_base=None):
        base=compose(config_name="train",overrides=["experiment=gpff_absorbing","globals.n_atom_basis=8","globals.n_interactions=1","globals.n_rbf=4"])
        model=hydra.utils.instantiate(base.model)
        source=tmp_path/"source_model"
        torch.save(model,source)
        cfg=compose(config_name="train",overrides=["experiment="+experiment,"globals.n_atom_basis=8","globals.n_interactions=1","globals.n_rbf=4","+init_weights="+str(source)])
        student=hydra.utils.instantiate(cfg.model)
        report=initialize_weights(student,source,allow_sigma=experiment.endswith("sigma"),report_path=tmp_path/"init.json")
        assert len(report["new_parameters"]) == (4 if experiment.endswith("sigma") else 0)
        if experiment.endswith("recovery"):
            task=hydra.utils.instantiate(cfg.task,model=student,
                optimizer_cls=torch.optim.AdamW,scheduler_cls=None)
            assert task.recovery.mode == "rollout"
            assert "recovery" in task.hparams
        # An arbitrary missing old parameter must never pass the allowlist.
        del student.output_modules[-1]
        with pytest.raises(ValueError,match="Unexpected initialization"):
            initialize_weights(student,source,allow_sigma=experiment.endswith("sigma"),report_path=tmp_path/"bad.json")


def test_study_report_retains_failed_attempts_and_excludes_nonphysical_histograms():
    from gpff.analysis.absorbing_report import reduce_records
    valid={"valid":True,"smiles":"novel","formula":[1,1,1,0,0],"numbers":[1,6,7],"size":3,"failure":None}
    failed={"valid":False,"smiles":"","formula":[0,0,0,0,0],"numbers":[99],"size":1,"failure":"unresolved_token"}
    metrics=reduce_records([valid,failed,valid],[[1,1,1,0,0]],set())
    assert metrics["attempts"] == 3
    assert metrics["validity"] == pytest.approx(2/3)
    assert metrics["uniqueness"] == .5
    assert metrics["vun"] == pytest.approx(1/3)
    assert metrics["all_output"]["excluded_nonphysical_atoms"] == 1
    assert metrics["all_output"]["molecule_presence"][2] == pytest.approx(2/3)
    assert metrics["all_output"]["element_shares"][2] == pytest.approx(1/3)
    assert metrics["valid_only"]["molecules"] == 2
    assert metrics["failures"] == {"unresolved_token":1}


def test_nonfinite_geometry_does_not_hide_overlapping_mask_and_absent_failures():
    from gpff.analysis.absorbing_report import failure_flags
    flags=failure_flags([{"nonfinite_geometry":True,"unresolved_mask":2,"absent":1,
                          "numbers":[99,99,0],"failure":"nonfinite_geometry"}])
    assert flags == {"nonfinite_geometry_molecules":1,"MASK_molecules":1,"MASK_atoms":2,
                     "ABSENT_molecules":1,"ABSENT_atoms":1}


def test_cached_full_neighbors_preserve_network_sampling_results():
    import copy
    import hydra
    from hydra import compose,initialize_config_dir
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]/"src/gpff/configs"
    with initialize_config_dir(config_dir=str(root),version_base=None):
        cfg=compose(config_name="train",overrides=["experiment=gpff_absorbing","globals.n_atom_basis=8","globals.n_interactions=1","globals.n_rbf=4"])
        model=hydra.utils.instantiate(cfg.model).eval()
    initial=batch(2,3)
    initial.pop(P.idx_i)
    initial.pop(P.idx_j)
    initial.update({P.idx:torch.arange(2),P.cell:torch.zeros(2,3,3),P.pbc:torch.zeros(2,3,dtype=torch.bool)})
    outputs=[]
    for cache in (False,True):
        torch.manual_seed(81)
        calc=Calculator(copy.deepcopy(model),device="cpu")
        state=snapshot(initial)
        if cache:
            state=calc.update_neighbors(calc.cast_batch(state))
        outputs.append(CoupledAbsorbingSampler(calculator=calc,n_steps=3).denoise(state))
    for key in (P.R,P.Z,keys.nfe):
        torch.testing.assert_close(outputs[0][key],outputs[1][key])


def test_summary_includes_final_confirmations_and_keeps_partition_labels(tmp_path):
    import importlib.util
    from pathlib import Path
    script=Path(__file__).resolve().parents[1]/"studies/absorbing_coupling/summarize.py"
    spec=importlib.util.spec_from_file_location("study_summary",script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for relative,partition in (("pilots5/control","val"),("confirmation5000/control/seed42069","test")):
        run=tmp_path/relative
        destination=run/"study_evaluation"
        destination.mkdir(parents=True)
        (run/"generation.json").write_text(json.dumps({"partition":partition,"mean_nfe":256,"wall_seconds":1}))
        (run/"manifest.json").write_text(json.dumps({"status":"completed","arguments":{"seed":42069}}))
        metric={"attempts":2,"validity":.5,"vun":.5,"uniqueness":1,"failures":{"radical":1},
            "molecule_bootstrap_95pct":{},"all_output":{"element_tv":.1,"element_shares":[.5,.3,.1,.1,0],
            "molecule_presence":[1,1,.5,.5,0],"n_o_cooccurrence":0}}
        (destination/"metrics.json").write_text(json.dumps(metric))
    output=tmp_path/"summary"
    module.summarize(tmp_path,output)
    rows=json.loads((output/"summary.json").read_text())
    assert {r["partition"] for r in rows} == {"val","test"}
    assert len(rows)==2
    groups=json.loads((output/"confirmation_by_seed.json").read_text())
    assert len(groups)==1 and groups[0]["n_generation_seeds"]==1
