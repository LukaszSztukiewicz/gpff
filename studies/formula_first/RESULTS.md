# Formula-first implementation and execution record

Updated 2026-10-30. Implementation and qualification are in progress. No full
training has been submitted yet; no final generation-quality claim is made.

Reference: **proven-ox-999**, `runs/default/26-10-28/proven-ox-999`, frozen epoch
5000, QM9 `b36d9cd218f5998d`, `gpff-py313-torch214-gcc.sif`. This is the user's
designated GPFF model without atom-type prediction. Its supplied atom types are
replaced by independently sampled training formulas in the new generator.

- Split SHA256: `560c21348398123394c7b974103c3ec9e079cf9d74dcacd0c8c4f4db8c76a7cb`.
- Reference epoch-5000 model SHA256: `632158c42522f7a8a7d2062c9e1d791d3fd76b77d380b489d6f6444006759d91`.
- Dataset SHA256: `527e3a65546f96aea3544e56ca8eb90828602940a3297af1f2a34796e4b8aad4`.
- Active frozen snapshot: `759d3d6-formula-first-v4` (proven-ox's source revision).
- Uploaded archive SHA256: `c7cfdd926fb64807370d0b02e600e2eb8477312c3049f7f3a99247f8568ca141`.
- Hydra study: `~/gpff/scratch/formula-first-20261030/`.

The earlier snapshot v1/check job 5151628 passed four formula tests and dry runs
but used fixed-snail as reference. Its artifacts remain under
`scratch/formula-first-20261029/`; they do not designate the baseline for this
study. The two reference splits happen to have identical SHA256 hashes. No old
jobs were cancelled and no existing run or dataset was changed.

| Job | Purpose | Recorded status |
|---|---|---|
| 5156925 | v2 CPU tests, proven-ox formula table, three dry-run configs | Completed, exit 0; four tests passed |
| 5156926 | v2 GPU qualification | Base training/export passed; generation exposed omitted-null-config bug |
| 5156940 | v2 structural evaluation | Obsolete dependency on failed v2 qualification; not cancelled |
| 5156963 | v3 CPU regression and dry runs | Five tests passed, including omitted config group |
| 5156964 / 5156965 | v3 qualification/evaluation | Qualification already running when superseded; evaluation held |
| 5156974 | v4 CPU regression and dry runs | Completed, exit 0; five tests passed |
| 5156975 | v4 three GPU short trainings and reference generation | Running after CPU checks |
| 5156976 | v4 pinned structural evaluator | Submitted after GPU checks |

The omitted-null-config bug was fixed by using `OmegaConf.select` for the
optional condition-generator group. v4 also rebases only the formula feature
onto `759d3d6`, preserving proven-ox's epoch-shuffle rule as well as its split
and training settings. Earlier snapshots and failed logs are retained.

The table contains 590 joint formulas fitted from 98,855 training molecules.
The dry-run configs pin proven-ox's split explicitly and retain its optimizer,
EMA, early-stopping policy and maximum epoch budget. The separate evaluation
wrapper is hashed as
`8439a383ce1fae63d96904ea583b42a18ccae9a34c29ad7dbd606399031dee88`.

The model architecture, losses, optimizer implementation and sampler kernel are
unchanged. Production integration consists of the reused empirical formula
prior, six-line Hydra prior config and small additions to `gpff_generate`.
All new training differences are configuration-only noise means (-0.7,-1.2,-0.2).
Full training uses proven-ox's 100000 maximum epochs and early stopping, with
fixed-milestone comparisons; Slurm allocates at most 48 hours per submission.

The local temporary worktree was cleared between interrupted sessions. It was
recovered from the byte-preserved Hydra archive into the persistent nested
worktree `.worktrees/formula-first`, now branch `feat/formula-first-proven`. Changes remain
uncommitted. See README.md for the derivation, design and evaluation protocol.
