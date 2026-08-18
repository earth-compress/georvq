| ID | CATEGORY | PURPOSE | INPUTS | PROMOTE |
|---|---|---|---|---|
| [1] | SKIP | Inspect GeoTessera API surface | none | no |
| [2] | SKIP | Inspect fetch_embedding and registry enumeration API | none | no |
| [3] | INFRA | Get registry fetch signature and total tile counts by year | none | no |
| [4] | SKIP | Fetch one Tessera tile raw + dequantized to inspect format | none | no |
| [5] | SKIP | Check nodata representation in coastal tile and GeoTessera c | none | no |
| [6] | INFRA | Verify smoke store round-trips through ChipStore and prep | store | no |
| [7] | SKIP | Compute AE store statistics from its index | store | no |
| [8] | DATA | AE store stats, tolerant of old manifest schema | store | no |
| [9] | SKIP | AE resident chips + tessera registry projections | store | no |
| [10] | INFRA | Count parameter distribution of current 2x model | none | no |
| [11] | INFRA | Count parameter distribution of current 2x model | none | no |
| [12] | INFRA | Parameter counts for 4x-latent candidates with trimmed trunk | none | yes |
| [13] | INFRA | Parameter counts for 4x-latent candidates with trimmed trunk | none | yes |
| [14] | SKIP | Narrow in on ~107M candidate | none | no |
| [15] | INFRA | Parameter counts for widened-tokenizer candidates | none | no |
| [16] | INFRA | Verify max-abs pinning makes magnitude recoverable from dire | store | no |
| [17] | DATA | Measure scale and vector-norm distribution on real chips | store | no |
| [18] | INFRA | Update manifest dequant note; find trainer PID | store | no |
| [19] | RATE | Kill trainer; scan sidecars for bad values and empty-crop ra | store | yes |
| [20] | SKIP | Reproduce the NaN with real pool and forward pass | store | no |
| [21] | TRAINDIAG | Reproduce NaN with no_grad forward passes | store | no |
| [22] | TRAINDIAG | Faithful training-loop repro hunting for NaN origin | store | no |
| [23] | TRAINDIAG | Faithful training-loop repro hunting for NaN origin | store | no |
| [24] | DATA | Scan all 83 scale sidecars for nan/inf/extreme values | store | no |
| [25] | DATA | Tail histogram of scale values across all sidecars | store | no |
| [26] | INFRA | Verify vq8 model build; check RAM and GPUs | none | no |
| [27] | QUANTIZER | Summarize spectrum, phase, gradient dependence | npz-cache | yes |
| [28] | MODELPERF | Split val cosine by fully-valid vs nodata-boundary tokens | checkpoint,store | yes |
| [29] | SKIP | Recompute error budget with correct annulus weighting | npz-cache | no |
| [30] | SKIP | Render diagnostic figure set | npz-cache | no |
| [31] | INFRA | Build and forward the 4-tier model on CPU | none | no |
| [32] | INFRA | Verify warm-start fix on CPU | checkpoint | no |
| [33] | MODELPERF | Spectral-shift ceiling estimate from diagnostic arrays | npz-cache | yes |
| [34] | MODELPERF | Measure recon power gain and spectral coherence vs signal | checkpoint,store | no |
| [35] | INFRA | Microbenchmark late_join 6 vs 0 at equal depth on GPU 1 | none | no |
| [36] | SKIP | Per-layer locality curves and tier-block averages | npz-cache | no |
| [37] | SKIP | Render AE attention figures | npz-cache | no |
| [38] | INFRA | Attention mass by distance, count-weighted | npz-cache | no |
| [39] | SKIP | Structure vs uniformity of far-field attention | npz-cache | no |
| [40] | SKIP | Per-sample far-field concentration in engaged encoder layers | checkpoint,store | no |
| [41] | INFRA | Full suite + build eff model | none | no |
| [42] | TRAINDIAG | Verify slim config builds; commit; check eff step time | logfile | no |
| [43] | SKIP | Bench eff model: ckpt and compile ablations | none | no |
| [44] | SKIP | Module-level timing breakdown of eff model | none | no |
| [45] | SKIP | Full suite + bench halo kernel | none | no |
| [46] | MODELPERF | Measure CPU-pool batch gather cost | store | no |
| [47] | INFRA | Verify patch4 single-vector AE builds | none | no |
| [48] | QUANTIZER | Measure effective latent rank per tier on s1_4x | checkpoint,store | yes |
| [49] | QUANTIZER | Effective latent rank per tier on the deployed AE model | checkpoint,store | yes |
| [50] | INFRA | Create and verify both bracket configs | none | no |
| [51] | INFRA | Create and verify AE probe trio configs | none | no |
| [52] | INFRA | Verify uf architecture builds | none | no |
| [53] | SKIP | Render page previews for visual check | none | no |
| [54] | SKIP | Find geographically overlapping AE/Tessera chip pairs | store | no |
| [55] | SKIP | Render co-located candidate pairs contact sheet | store | no |
| [56] | SKIP | Cross-correlate structure maps to find true AE/Tessera offse | store | no |
| [57] | SKIP | Render aligned co-located pair for verification | store | no |
| [58] | SKIP | Fix shape bug, render aligned pair | store | no |
| [59] | INFRA | Check RAM, create and verify w32hd config | none | no |
| [60] | INFRA | Create uf stage-2 config, verify build and warm-start | checkpoint | no |
| [61] | QUANTIZER | PCA of a global random Tessera pixel sample | store | no |
| [62] | RATE | Neighbor-predictive entropy (rANS-achievable rate) on Tesser | store,logfile | yes |
| [63] | DATA | Measure codeword norm shrinkage vs residual norm per level | checkpoint,store | no |
| [64] | QUANTIZER | Measure mean-vs-variance dominance in the latents per tier | checkpoint,store | yes |
| [65] | QUANTIZER | Effective rank of the uf encoder's latents per tier | checkpoint,store | yes |
| [66] | QUANTIZER | Track codebook degradation from k-means init over EMA steps | checkpoint,store,logfile | yes |
| [67] | TRAINDIAG | Test whether decay rate or bf16 scores drive the degradation | checkpoint,store | yes |
| [68] | QUANTIZER | Directly compare bf16 vs fp32 assignment on trained codebook | checkpoint,store,logfile | yes |
| [69] | QUANTIZER | Test whether k-means sample size explains the low utilizatio | checkpoint,store,logfile | yes |
| [70] | DATA | Compare codebook utilization in raw vs stage-1 normalized ge | checkpoint,store,logfile | yes |
| [71] | QUANTIZER | Distortion vs codebook size: is K=16384 actually buying anyt | checkpoint,store,logfile | yes |
| [72] | QUANTIZER | Export train/holdout latent vector store with window-disjoin | checkpoint,store | yes |
| [73] | SKIP | Estimate intrinsic dimensionality of the challenge vectors | logfile | no |
| [74] | SKIP | Re-estimate intrinsic dimension controlling for within-windo | logfile | no |
| [75] | QUANTIZER | Independently re-score the winning codebooks in fp64 | logfile | yes |
| [76] | SKIP | Test whether local blur is isotropic via difference-vector s | none | no |
| [77] | QUANTIZER | Measure within-window vs between-window latent variance per  | checkpoint,store | yes |
| [78] | SKIP | Check whether fine-tier collapse is specific to the 4-tier a | checkpoint,store | no |
| [79] | SKIP | Repeat K-saturation test on a healthy fine tier | checkpoint,store,logfile | no |
| [80] | QUANTIZER | Check w32hd tier health and fine-tier codebook utilization | checkpoint,store,logfile | yes |
| [81] | QUANTIZER | Compare latent norm spread on healthy vs collapsed tiers | checkpoint,store | yes |
| [82] | SKIP | Final w32hd numbers and tier health check | checkpoint,store,logfile | no |
| [83] | INFRA | Get exact encoder/decoder parameter split for w32hd | checkpoint | no |
| [114] | RATE | Fill low-rate gap in AE-only arm | logfile,npz-cache | yes |
| [115] | TRAINDIAG | Fetch wandb loss series around the spike | wandb | no |
| [116] | RUNCOMPARE | Check whether the 2100 spike is unique | wandb | no |
| [117] | DATA | Check tessera scale tail | store | no |
| [118] | RUNCOMPARE | Check whether relative error spiked too | wandb | no |
| [119] | INFRA | Check initialized flags in saved checkpoints | checkpoint | no |
| [120] | TRAINDIAG | Quantify loss progress per step and per sample across the ba | wandb,logfile | no |
| [121] | RUNCOMPARE | Compare ceiling metrics against k256 | wandb | yes |
| [122] | RUNCOMPARE | Get fine-grained view of the collapse | wandb | no |
| [123] | RUNCOMPARE | Check grad norms in the clipped run | wandb,logfile | no |
| [124] | RUNCOMPARE | Inspect grad norm trace | wandb | no |
| [125] | RUNCOMPARE | Inspect grad norms through the danger region | wandb | no |
| [126] | RUNCOMPARE | Fetch k256 centred-ceiling trajectory for comparison | wandb | yes |
| [127] | RUNCOMPARE | Check grad norms around the spike | wandb | no |
| [128] | RUNCOMPARE | Locate the clipped spike | wandb | no |
| [129] | RUNCOMPARE | Compare ceilings across all 64px tessera runs | wandb | yes |
| [130] | RUNCOMPARE | Inspect wandb config structure | wandb | no |
| [131] | RUNCOMPARE | Diagnose the run filter | wandb | no |
| [132] | RUNCOMPARE | Check exact window/source values | wandb | no |
| [133] | RUNCOMPARE | Compare ceilings across 64px tessera runs, fetching full con | wandb | yes |
| [134] | RUNCOMPARE | Compare throughput across runs and their efficiency settings | wandb | yes |
| [135] | RUNCOMPARE | Find the _eff runs in wandb | wandb | no |
| [136] | RUNCOMPARE | Check per-tier codebook usage after init | wandb | yes |
| [137] | RUNCOMPARE | Check whether coarse-only ccos is rising | wandb,logfile | no |
| [138] | RATE | Check per-band DN==0 rates in S2 | none | yes |
| [139] | RUNCOMPARE | Check coarse-tier recovery through step 17000 | wandb,logfile | no |
| [140] | RUNCOMPARE | Check the coarse-tier pickup at 18k | wandb,logfile | no |
| [141] | SKIP | Measure magnitude spread across windows | npz-cache | no |
| [142] | RUNCOMPARE | Check the magnitude-aware error metric | wandb | no |
| [143] | RUNCOMPARE | Check the coarse tier at 3k into the resume | wandb,logfile | no |
| [144] | RUNCOMPARE | Compare coarse tier against the k256 baseline | wandb | yes |
| [145] | QUANTIZER | Compute unquantized latent rate per tier | logfile | yes |
| [146] | RUNCOMPARE | Compute relative gap-closure trends per tuple | wandb | no |
| [147] | SKIP | Test filter thresholds to size the agent task | none | no |
