"""GPU performance audit. Execute only inside Apptainer on Hydra via Slurm.

Historical component comparisons run with fixed HEAD dependencies; endpoint
workloads run in separate processes with their complete historical PYTHONPATH.
All reported times include CUDA synchronization at the measurement boundaries.
"""
import argparse
import contextlib
import gc
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import sqlite3
import statistics
import sys
import time
import traceback

import numpy as np
import torch
from schnetpack import properties as P
from schnetpack.atomistic import PairwiseDistances
from schnetpack.data.loader import _atoms_collate_fn
from schnetpack.model import NeuralNetworkPotential
from schnetpack.nn.cutoff import CosineCutoff
from schnetpack.nn.radial import GaussianRBF
from gpff import keys as K
from gpff.model.heads import GatedEquivariant
from gpff.sample.base import Calculator
from gpff.sample.direct_denoising import DirectDenoisingSampler
from gpff.utils.neighbors import compute_neighbors

ROOT = Path(os.environ.get("AUDIT_ROOT", "/home/lsztuk/gpff/scratch/speedup-audit-20261029"))
CODE = Path("/home/lsztuk/gpff/frozen_code/speedup-audit-20261029")
DB = Path("/home/lsztuk/gpff/data/qm9/b36d9cd218f5998d/qm9_filtered.db")
SPLIT = DB.parent / "split.npz"
ROWS = []


def emit(obj):
    print(json.dumps(obj, default=str), flush=True)
    ROWS.append(obj)


def history(sha, relative):
    name = "audit_" + sha + "_" + relative.replace("/", "_").replace(".", "_")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, CODE / sha / "src/gpff" / relative)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sync():
    torch.cuda.synchronize()


def clone(batch, device=None):
    return {k: (v.detach().clone().to(device) if device else v.detach().clone())
            if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def data(n=512):
    path = ROOT / "inputs.pt"
    if path.exists():
        molecules = torch.load(path, weights_only=False)
    else:
        rng = np.random.default_rng(999)
        split = np.load(SPLIT)
        train_key = next(k for k in split.files if "train" in k)
        chosen = rng.choice(split[train_key], size=512, replace=False)
        molecules = []
        with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as db:
            for j, idx in enumerate(chosen):
                numbers, positions, na = db.execute(
                    "select numbers, positions, natoms from systems where id=?", (int(idx)+1,)
                ).fetchone()
                molecules.append({
                    P.idx: torch.tensor([int(idx)]), P.n_atoms: torch.tensor([na]),
                    P.Z: torch.from_numpy(np.frombuffer(numbers, dtype=np.int32).astype(np.int64)),
                    P.R: torch.from_numpy(np.frombuffer(positions, dtype=np.float64).copy().reshape(na, 3)),
                    P.cell: torch.zeros(1, 3, 3, dtype=torch.float64),
                    P.pbc: torch.zeros(3, dtype=torch.bool),
                })
        torch.save(molecules, path)
    return molecules[:n], _atoms_collate_fn(molecules[:n])


def model(painn_cls, width=256):
    torch.manual_seed(999)
    return NeuralNetworkPotential(
        representation=painn_cls(n_atom_basis=width, n_interactions=4,
            radial_basis=GaussianRBF(n_rbf=600, cutoff=150.0),
            cutoff_fn=CosineCutoff(150.0), return_vector_representation=True),
        input_modules=[PairwiseDistances()],
        output_modules=[GatedEquivariant(n_in=width, n_layers=3)],
    ).to(device="cuda", dtype=torch.float32)


def paired(label, before, after, n=7, inner=1, warm=2, metadata=None):
    for fn in (before, after):
        for _ in range(warm):
            fn()
        sync()
    measurements = [[], []]
    for i in range(n):
        order = (0, 1) if i % 2 == 0 else (1, 0)
        for side in order:
            fn = (before, after)[side]
            sync()
            t = time.perf_counter()
            for _ in range(inner):
                fn()
            sync()
            measurements[side].append((time.perf_counter()-t)*1000/inner)
    ratios = np.array(measurements[0]) / measurements[1]
    rng = np.random.default_rng(991)
    boot = np.median(rng.choice(ratios, (10000, n)), axis=1)
    emit({"test": label, "before_ms": statistics.median(measurements[0]),
          "after_ms": statistics.median(measurements[1]),
          "speedup": float(np.median(ratios)),
          "paired_speedup_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
          "before_samples_ms": measurements[0], "after_samples_ms": measurements[1],
          **(metadata or {})})


def single(label, fn, n=7, inner=1, warm=2, metadata=None):
    sync()
    cold_start=time.perf_counter()
    fn()
    sync()
    cold_ms=(time.perf_counter()-cold_start)*1000
    for _ in range(warm):
        fn()
    sync()
    values=[]
    torch.cuda.reset_peak_memory_stats()
    for _ in range(n):
        sync()
        t=time.perf_counter()
        for _ in range(inner):
            fn()
        sync()
        values.append((time.perf_counter()-t)*1000/inner)
    emit({"test":label, "median_ms":statistics.median(values), "samples_ms":values,
          "first_call_ms":cold_ms,
          "peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30, **(metadata or {})})


def setup(args):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(getattr(args,"deterministic",False))
    torch.manual_seed(999)
    np.random.seed(999)
    random.seed(999)
    emit({"environment": {"gpu":torch.cuda.get_device_name(), "torch":torch.__version__,
          "cuda":torch.version.cuda, "job":os.getenv("SLURM_JOB_ID"),
          "node":os.uname().nodename, "threads":torch.get_num_threads(),
          "matmul_precision":torch.get_float32_matmul_precision(),
          "deterministic":torch.are_deterministic_algorithms_enabled(),
          "pythonpath":os.getenv("PYTHONPATH"), "args":vars(args)}})


def components(args):
    molecules, cpu = data(128)
    gpu = {k: v.to("cuda", dtype=torch.float32) if v.is_floating_point() else v.cuda()
           for k,v in cpu.items()}
    graph = compute_neighbors(gpu, fully_connected=True, device="cuda")
    n = len(graph[P.n_atoms])
    graph[K.converged] = torch.zeros(n, dtype=torch.bool, device="cuda")
    graph[K.nfe] = torch.zeros(n, dtype=torch.int32, device="cuda")

    # ef74aa3: repeated std estimation of the same molecule layout.
    old = history("759d3d6", "sample/base.py")
    old._pad_by_molecule = history("759d3d6", "utils/alignment.py")._pad_by_molecule
    new = history("ef74aa3", "sample/base.py")
    old_calc, new_calc = old.Calculator(), new.Calculator()
    force=torch.randn_like(graph[P.R])
    paired("ef74aa3/std_per_step", lambda:old_calc._compute_std_per_molecule(graph[P.idx_m],force),
           lambda:new_calc._compute_std_per_molecule(graph[P.idx_m],force), inner=20)

    # b24eda8: batched covariance; both CPU-prior and GPU-step paths matter.
    old=history("ef74aa3", "utils/alignment.py")
    new=history("b24eda8", "utils/alignment.py")
    for device,b in (("cpu",cpu),("cuda",gpu)):
        cov=torch.eye(3,dtype=b[P.R].dtype,device=device).expand(n,3,3).contiguous()
        center=torch.zeros(n,3,dtype=b[P.R].dtype,device=device)
        f=lambda m:m.batch_apply_covariances(b[P.R],b[P.idx_m],cov,centers=center)
        a,z=f(old),f(new)
        emit({"correctness":"b24eda8/"+device,"max_abs_diff":(a-z).abs().max().item()})
        paired("b24eda8/covariance_"+device,lambda:f(old),lambda:f(new),inner=2)

    # 93cca41 only replaces compute_neighbors, not the training data transform.
    old=history("b24eda8", "utils/neighbors.py")
    new=history("93cca41", "utils/neighbors.py")
    for device,b in (("cpu",cpu),("cuda",gpu)):
        f=lambda m:m.compute_neighbors(b,fully_connected=True,device=device)
        a,z=f(old),f(new)
        emit({"correctness":"93cca41/"+device,
              "all_keys_equal":all(torch.equal(a[k],z[k]) for k in a)})
        paired("93cca41/neighbors_"+device,lambda:f(old),lambda:f(new),inner=2)

    # Result metadata, Atoms construction, recorder, and final D2H clone.
    features=clone(graph)
    na=len(features[P.R])
    features["scalar_representation"]=torch.randn(na,256,device="cuda")
    features["vector_representation"]=torch.randn(na,3,256,device="cuda")
    features[K.forces]=torch.randn(na,3,device="cuda")
    features[K.sigma_curr]=torch.ones(na,1,device="cuda")
    from gpff.sample.results import clone_batch
    host=clone_batch(features)
    old=history("93cca41","sample/results.py")
    old.batch_to_atoms_list=history("93cca41","utils/batching.py").batch_to_atoms_list
    new=history("4839f63","sample/results.py")
    new.batch_to_atoms_list=old.batch_to_atoms_list
    paired("4839f63/serialize_CPU",lambda:old.batches_to_relaxed([host]),
           lambda:new.batches_to_relaxed([host]), n=7, warm=1)

    old_b=history("a0e6c9b","utils/batching.py")
    new_b=history("697f0f6","utils/batching.py")
    paired("697f0f6/atoms_CPU",lambda:old_b.batch_to_atoms_list(host),
           lambda:new_b.batch_to_atoms_list(host),inner=3)
    old_obs=history("a0e6c9b","sample/observers.py").TrajectoryRecorder()
    new_obs=history("697f0f6","sample/observers.py").TrajectoryRecorder()
    def record(obs):
        obs.snapshots=[]
        obs.on_step_end(features,1)
    paired("697f0f6/record_GPU",lambda:record(old_obs),lambda:record(new_obs),inner=3)

    old=history("632660a","sample/results.py")
    new=history("8991e5f","sample/results.py")
    for device,b in (("cpu",host),("cuda",features)):
        paired("8991e5f/clone_"+device,lambda:old.clone_batch(b),
               lambda:new.clone_batch(b),inner=5)

    # GMM uses the same fitted mixtures on both sides; fitting is separately timed.
    old=history("4839f63","sample/shape_predictor.py")
    new=history("24a2ce2","sample/shape_predictor.py")
    from sklearn.mixture import GaussianMixture
    mixtures={}
    rng=np.random.default_rng(999)
    for size in cpu[P.n_atoms].unique().tolist():
        mixtures[size]=GaussianMixture(n_components=5).fit(rng.normal(size=(100,3)))
    gm=[]
    for mod in (old,new):
        obj=mod.ShapeGMM.__new__(mod.ShapeGMM)
        obj.models=mixtures
        obj.variance_floor=1e-6
        obj.n_components=5
        obj.verbose=False
        gm.append(obj)
    for device,b in (("cpu",cpu),("cuda",gpu)):
        paired("24a2ce2/GMM_draw_"+device,lambda:gm[0].sample_from_batch(b),
               lambda:gm[1].sample_from_batch(b),inner=2)
    # Include dataset traversal and mixture fitting on 512 real training molecules.
    fitdata,_=data(512)
    def fit(obj):
        np.random.seed(999)
        obj.fit(fitdata)
    paired("24a2ce2/GMM_fit_512_CPU",lambda:fit(gm[0]),lambda:fit(gm[1]),n=3,warm=1)

    old=history("38cb9db","sample/priors.py").ScaffoldGaussianPrior(sigma=30.)
    new=history("47efdc3","sample/priors.py").ScaffoldGaussianPrior(sigma=30.)
    for device,b in (("cpu",cpu),("cuda",gpu)):
        mask=torch.arange(len(b[P.R]),device=device)%2==0
        # Guarantee at least one scaffold per molecule, including single atoms.
        offsets=b[P.n_atoms].cumsum(0)-b[P.n_atoms]
        mask[offsets]=True
        conditions={K.scaffold_mask:mask}
        paired("47efdc3/scaffold_prior_"+device,
               lambda:old.sample_from_batch(clone(b),conditions),
               lambda:new.sample_from_batch(clone(b),conditions),inner=2)

    # CPU training data alignment (run in DataLoader workers, never on GPU).
    old=history("697f0f6","utils/alignment.py")
    new=history("3b17abc","utils/alignment.py")
    def align(mod):
        for mol in molecules[:64]:
            pos=mol[P.R]
            mod.align_permutation(pos+0.2,pos,mol[P.Z],mol[P.Z])
    paired("3b17abc/align_64_CPU",lambda:align(old),lambda:align(new),inner=2)

    # Per-step bookkeeping separately from the expensive neural network.
    old=history("3c78267","sample/base.py")
    new=history("831b100","sample/base.py")
    def bookkeeping(mod):
        obj=object.__new__(mod.Sampler)
        # ABC cannot be allocated; use its methods with a duck-typed self.
    class Dummy: pass
    obj=Dummy()
    b=clone(graph)
    updates={P.R:torch.randn_like(b[P.R]), K.sigma_curr:torch.ones(na,1,device="cuda")}
    b[K.sigma_curr]=updates[K.sigma_curr].clone()
    def step(mod):
        mod.Sampler.update_batch(obj,b,updates)
        mod.Sampler.check_convergence(obj,b,2)
    paired("831b100/bookkeeping_step",lambda:step(old),lambda:step(new),inner=30)

    # EMA has the real default architecture's parameter tensors.
    from gpff.model.painn import PaiNN
    from torch_ema import ExponentialMovingAverage
    from gpff.callbacks import _ForeachEMA
    net=model(PaiNN)
    old=ExponentialMovingAverage(net.parameters(),decay=.995)
    new=_ForeachEMA(net.parameters(),decay=.995)
    paired("d464b76/EMA_GPU",old.update,new.update,inner=20)
    emit({"correctness":"d464b76/EMA","all_equal":all(torch.equal(a,b) for a,b in zip(old.shadow_params,new.shadow_params))})


def neural(args):
    """Paired historical PaiNN changes, with fresh and reused neighbor tensors."""
    _,cpu=data(args.batch)
    graph=compute_neighbors(cpu,fully_connected=True,device="cuda")
    graph={k:v.float() if v.is_floating_point() else v for k,v in graph.items()}
    graph[K.converged]=torch.zeros(args.batch,device="cuda",dtype=torch.bool)
    graph[K.nfe]=torch.zeros(args.batch,device="cuda",dtype=torch.int32)
    pairs=[("38cb9db","d464b76","38cb9db"),("a0e6c9b","831b100","a0e6c9b")]
    if args.compile:
        pairs=[("ea093f3","3b17abc","ea093f3")]
    for label,before,after in pairs:
        mods=[history(s,"model/painn.py") for s in (before,after)]
        nets=[model(m.PaiNN,args.width) for m in mods]
        # Identical weights even if constructor consumption changes.
        nets[1].load_state_dict(nets[0].state_dict())
        if args.compile:
            for net in nets:
                net.compile(dynamic=True)
        def forward(i,fresh=False):
            b=dict(graph)
            if fresh:
                b[P.idx_i]=graph[P.idx_i].clone()
                b[P.idx_j]=graph[P.idx_j].clone()
            return nets[i](b)[K.forces]
        with torch.no_grad():
            cold=[]
            outputs=[]
            for i in range(2):
                sync()
                t=time.perf_counter()
                outputs.append(forward(i))
                sync()
                cold.append((time.perf_counter()-t)*1000)
            a,z=outputs
            emit({"first_forward_ms":cold,"test":label+"/cold","compiled":args.compile})
            emit({"correctness":label+"/forward","max_abs_diff":(a-z).abs().max().item(),
                  "relative_l2":((a-z).norm()/a.norm()).item()})
            paired(label+"/forward_reused",lambda:forward(0),lambda:forward(1),inner=3,
                   metadata={"batch":args.batch,"width":args.width,"compiled":args.compile})
            paired(label+"/forward_fresh",lambda:forward(0,True),lambda:forward(1,True),inner=3,
                   metadata={"batch":args.batch,"width":args.width,"compiled":args.compile})
        if args.train:
            target=torch.randn_like(graph[P.R])
            opts=[torch.optim.Adam(net.parameters(),lr=1e-4) for net in nets]
            def train(i):
                opts[i].zero_grad(set_to_none=True)
                pred=forward(i,True)
                loss=(pred-target).square().mean()
                loss.backward()
                opts[i].step()
            paired(label+"/train_fresh",lambda:train(0),lambda:train(1),inner=2,
                   metadata={"batch":args.batch,"width":args.width,"compiled":args.compile})
        del nets
        if args.train:
            del opts
        gc.collect()
        torch.cuda.empty_cache()


def workload(args):
    """Whole-revision training steps / direct sampling, with no monkey patches."""
    from gpff.model.painn import PaiNN
    from gpff.sample.results import batches_to_relaxed
    _,cpu=data(max(args.batch,128))
    _,cpu_sample=data(128)
    net=model(PaiNN,args.width)
    if args.compile:
        net.compile(dynamic=True)
    if args.train:
        _,traincpu=data(args.batch)
        train=compute_neighbors(traincpu,fully_connected=True,device="cuda")
        train={k:v.float() if v.is_floating_point() else v for k,v in train.items()}
        torch.manual_seed(998)
        target=torch.randn_like(train[P.R])
        train[P.R]=train[P.R]+target/2
        from gpff.task import loss_fn
        weights=torch.ones_like(target)
        opt=torch.optim.Adam(net.parameters(),lr=1e-4)
        try:
            from gpff.callbacks import _ForeachEMA
            ema=_ForeachEMA(net.parameters(),decay=.995)
        except ImportError:
            from torch_ema import ExponentialMovingAverage
            ema=ExponentialMovingAverage(net.parameters(),decay=.995)
        def step():
            b=dict(train)
            b[P.idx_i]=train[P.idx_i].clone()
            b[P.idx_j]=train[P.idx_j].clone()
            opt.zero_grad(set_to_none=True)
            loss=loss_fn(net(b)[K.forces],target,weights)
            loss.backward()
            opt.step()
            ema.update()
        single("workload/train",step,inner=2,warm=3,metadata={"batch":args.batch,"width":args.width,"compiled":args.compile})
    net.eval()
    calc=Calculator(model=net,device="cuda",dtype=torch.float32)
    sampler=DirectDenoisingSampler(calculator=calc,n_steps=args.steps,stochastic_lambda=0.,f_max=0.)
    def sample():
        # Match gpff_generate: CPU prior, CPU->GPU casting and fresh neighbor list.
        b=clone(cpu_sample)
        torch.manual_seed(42069)
        b[P.R]=torch.randn_like(b[P.R])*30.
        result=sampler.denoise(b)
        if args.serialize:
            batches_to_relaxed([result])
        return result
    with torch.no_grad():
        single("workload/sample",sample,n=5,warm=1,metadata={"batch":128,"width":args.width,"steps":args.steps,"compiled":args.compile,"serialize":args.serialize})
        out=sample()
        emit({"sample_check":{"finite":torch.isfinite(out[P.R]).all().item(),
              "nfe_mean":out[K.nfe].float().mean().item(),"nfe_max":out[K.nfe].max().item()}})


def converged(args):
    from gpff.model.painn import PaiNN
    net=model(PaiNN,args.width).eval()
    _,cpu=data(args.batch)
    graph=compute_neighbors(cpu,fully_connected=True,device="cuda")
    graph={k:v.float() if v.is_floating_point() else v for k,v in graph.items()}
    graph[K.nfe]=torch.zeros(args.batch,dtype=torch.int32,device="cuda")
    old=history("47efdc3","sample/base.py")
    new=history("3c78267","sample/base.py")
    calcs=[old.Calculator(model=net),new.Calculator(model=net)]
    with torch.no_grad():
        for fraction in (0.,.25,.5,.75,.9):
            graph[K.converged]=torch.arange(args.batch,device="cuda")<int(args.batch*fraction)
            def forward(i):
                return calcs[i].calculate(graph,forces=True)
            paired("3c78267/predict_converged_"+str(fraction),lambda:forward(0),lambda:forward(1),inner=3,
                   metadata={"batch":args.batch,"width":args.width,"converged_fraction":fraction})


def deferred(args):
    """Actual ASE DB writing plus RDKit checks, with final queue flush included."""
    from ase.db import connect
    from gpff.cli.generate import _store_samples
    from gpff.analysis import get_rdkit_validity
    from gpff.model.painn import PaiNN
    from gpff.sample.observers import DeferredWork
    from gpff.sample.results import batches_to_relaxed
    net=model(PaiNN,args.width).eval()
    if args.compile:
        net.compile(dynamic=True)
    _,cpu=data(128)
    calc=Calculator(model=net)
    sampler=DirectDenoisingSampler(calculator=calc,n_steps=args.steps,stochastic_lambda=0.,f_max=0.)
    # Valid training geometries for the queued progress signal: the untrained
    # model can make pathological structures, which is not a representative RDKit cost.
    metadata=clone(cpu)
    metadata[K.nfe]=torch.full((128,),args.steps,dtype=torch.int32)
    metadata[K.converged]=torch.ones(128,dtype=torch.bool)
    samples=batches_to_relaxed([metadata])
    counter=[0]
    def run(overlap):
        counter[0]+=1
        path=str(ROOT/f"write-{os.getenv('SLURM_JOB_ID')}-{counter[0]}.db")
        work=DeferredWork()
        with connect(path) as db:
            db.metadata={"audit":True}
        for _ in range(3):
            b=clone(cpu)
            b[P.R]=torch.randn_like(b[P.R])*30
            sampler.denoise(b,observers=[work] if overlap else [])
            work.finish()
            work.add(*_store_samples(path,samples,128))
            if not overlap:
                work.finish()
        work.finish()
        with connect(path) as db:
            assert len(db)==384
    with torch.no_grad():
        paired("632660a/3_batch_pipeline",lambda:run(False),lambda:run(True),n=5,warm=1,
               metadata={"batch":128,"width":args.width,"steps":args.steps,"batches":3,"compiled":args.compile,"rdkit":"real QM9 geometries"})


def extra(args):
    from gpff.model.painn import PaiNN
    from gpff.sample.base import Sampler
    _,cpu=data(128)
    net=model(PaiNN,args.width).eval()
    calc=Calculator(model=net)
    sampler=DirectDenoisingSampler(calculator=calc,n_steps=16,stochastic_lambda=0.,f_max=0.)
    old=history("3c78267","sample/base.py")
    new=history("831b100","sample/base.py")
    original=(Sampler.update_batch,Sampler.check_convergence)
    def sample(mod):
        Sampler.update_batch=mod.Sampler.update_batch
        Sampler.check_convergence=mod.Sampler.check_convergence
        b=clone(cpu)
        torch.manual_seed(42069)
        b[P.R]=torch.randn_like(b[P.R])*30
        return sampler.denoise(b)
    with torch.no_grad():
        try:
            paired("831b100/full_DD16",lambda:sample(old),lambda:sample(new),n=7,warm=1)
        finally:
            Sampler.update_batch,Sampler.check_convergence=original
    old=history("ef74aa3","utils/alignment.py")
    new=history("b24eda8","utils/alignment.py")
    for geometry in ("real","random"):
        pos=cpu[P.R].cuda().float()
        if geometry=="random":
            torch.manual_seed(123)
            pos=torch.randn_like(pos)
        idx=cpu[P.idx_m].cuda()
        cov=torch.eye(3,device='cuda').expand(128,3,3).contiguous()
        center=torch.zeros(128,3,device='cuda')
        reference=old.batch_apply_covariances(pos.double(),idx,cov.double(),centers=center.double()).float()
        for precision in ("high","highest"):
            torch.set_float32_matmul_precision(precision)
            a=old.batch_apply_covariances(pos,idx,cov,centers=center)
            b=new.batch_apply_covariances(pos,idx,cov,centers=center)
            emit({"correctness":"b24eda8/precision","geometry":geometry,"precision":precision,
                  "old_new_maxabs":(a-b).abs().max().item(),
                  "old_reference_maxabs":(a-reference).abs().max().item(),
                  "new_reference_maxabs":(b-reference).abs().max().item(),
                  "old_new_relative_l2":((a-b).norm()/reference.norm()).item()})
    torch.set_float32_matmul_precision("high")


def adaptive(args):
    from schnetpack.utils import load_model
    from gpff.sample.ancestral_sampling import AdaptiveAncestralSampler
    from gpff.sample.observers import SamplerObserver
    checkpoint='/home/lsztuk/gpff/runs/default/26-10-26/seeded-monkey-1e4-basis-512/frozen/best_model'
    net=load_model(checkpoint,device='cpu').cuda().float().eval()
    _,cpu=data(128)
    class Counts(SamplerObserver):
        def on_start(self,sampler,batch): self.counts=[]
        def on_step_end(self,batch,step): self.counts.append(int((~batch[K.converged]).sum()))
    calcs=[Calculator(model=net,skip_converged=flag) for flag in (False,True)]
    samplers=[AdaptiveAncestralSampler(calculator=c,n_steps=128,n_steps_target=16) for c in calcs]
    last={}
    def run(i):
        b=clone(cpu)
        torch.manual_seed(42069)
        b[P.R]=torch.randn_like(b[P.R])*30.
        count=Counts()
        out=samplers[i].denoise(b,observers=[count])
        last[i]=(out,count.counts)
        return out
    with torch.no_grad():
        paired("3c78267/real_adaptive_ancestral",lambda:run(0),lambda:run(1),n=5,warm=1,
               metadata={"checkpoint":checkpoint,"batch":128,"width":512,"max_steps":128,"target_steps":16})
    a,z=last[0][0],last[1][0]
    emit({"correctness":"3c78267/real_adaptive","positions_maxabs":(a[P.R]-z[P.R]).abs().max().item(),
          "nfe_equal":torch.equal(a[K.nfe],z[K.nfe]),"nfe_mean_before":a[K.nfe].float().mean().item(),
          "nfe_mean_after":z[K.nfe].float().mean().item(),
          "active_counts_before":last[0][1],"active_counts_after":last[1][1]})
    with torch.no_grad():
        repeat=run(0)
    emit({"correctness":"3c78267/repeat_without_skipping",
          "positions_maxabs":(a[P.R]-repeat[P.R]).abs().max().item(),
          "nfe_equal":torch.equal(a[K.nfe],repeat[K.nfe])})
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    with torch.no_grad():
        strict_a,strict_b=run(0),run(1)
    emit({"correctness":"3c78267/highest_deterministic",
          "positions_maxabs":(strict_a[P.R]-strict_b[P.R]).abs().max().item(),
          "nfe_equal":torch.equal(strict_a[K.nfe],strict_b[K.nfe]),
          "nfe_mean_before":strict_a[K.nfe].float().mean().item(),
          "nfe_mean_after":strict_b[K.nfe].float().mean().item()})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["components","neural","workload","converged","deferred","extra","adaptive"])
    parser.add_argument("--batch",type=int,default=128)
    parser.add_argument("--width",type=int,default=256)
    parser.add_argument("--steps",type=int,default=16)
    parser.add_argument("--compile",action="store_true")
    parser.add_argument("--train",action="store_true")
    parser.add_argument("--serialize",action="store_true")
    parser.add_argument("--deterministic",action="store_true")
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    setup(args)
    try:
        globals()[args.mode](args)
    except Exception:
        emit({"error":traceback.format_exc()})
        raise
    finally:
        Path(args.out).write_text(json.dumps(ROWS,indent=2,default=str)+"\n")


if __name__=="__main__":
    main()
