# Auditing a binary offloading decision on a fixed pool

The unit is a predeclared frame. For each of the N frames, fix the local and backend binary decisions, the operational cutoffs, and one binary reference label. Let D count the frames for which the deployed decisions differ. Among them, let R count the cases where the backend agrees with the reference. The backend's accuracy minus local accuracy on these N frames is

$$
\Delta=(2R-D)/N.
$$

Both classifiers incur the same zero-one loss on an agreement. On a disagreement exactly one classifier is correct, contributing +1 or −1 to the signed numerator. This cancellation requires two binary actions and a binary reference. A multiclass disagreement can leave both classifiers wrong. A three-state workflow with abstentions needs an explicit utility and a corresponding outcome model.

## Invert the finite-population tails

Select n disagreement indices uniformly without replacement before inspecting their reference labels. If x of the labels favor the backend, then for any candidate r for R,

$$
P_r(X=t)=\binom{r}{t}\binom{D-r}{n-t}/\binom{D}{n}.
$$

Use the integer confidence set containing all feasible r for which both inclusive tail probabilities, evaluated at x, exceed 0.025. The inclusion inequality is strict; tail equality excludes r. Monotonicity in r makes the result an integer interval [L,U]. Its two-sided coverage is at least 95% for each fixed value of R, since either erroneous tail rejection occurs with probability at most 2.5%. The interval for Δ is [(2L−D)/N,(2U−D)/N]. Report a positive direction if 2L>D, a negative direction if 2U<D, and an undetermined direction otherwise. For D=0 the pool difference is exactly zero. For n=0 with D>0 both extremes remain possible; a full disagreement census recovers R exactly.

Arbitrary correlation among the fixed frame labels, repeated plants, or repeated capture sessions does not change this randomization calculation. The sample must be uniform among the preidentified D windows and independent of their unknown labels. The claim concerns this finite pool. Transfer to a fresh orchard, future season, wider set of subjects, or differently selected population needs separate evidence. Biased label availability, noisy reference labels, changing policies after inspecting outcomes, and repeatedly looking until certification require additional assumptions or different designs.

## Equal manual-label budget

Let b be the maximum number of labels, with b ≤ N. A targeted audit buys min(b,D) disagreement labels. A full-frame simple random audit buys b labels; its number K of disagreements obeys

$$
P(K=k)=\binom Dk\binom{N-D}{b-k}/\binom Nb .
$$

Conditional on K=k, the observed number of corrections has exactly the same hypergeometric law as a targeted audit of k disagreements. Consequently the full-frame design's exact probability of certifying the correct strict direction is the weighted sum of targeted k-label probabilities. For the historical 60-image hardware cohort, D=R=1. At b=1 the targeted design discovers the unique correction with probability one; the full-frame design does so with probability 1/60. At a fixed b<60 the latter probability is b/60. The case is illustrative because its references are already available.

In an information-theoretic comparison of these sampling designs, the larger targeted sample can emulate the full-frame label evidence relevant to Δ. Draw K with the law above independently and uniformly discard labels until K remain. This construction does not imply a uniformly higher rejection probability for one fixed discrete equal-tailed procedure. At N=9,D=8,R=5,b=5 the targeted sign power is 1/56, while full-frame sampling achieves 1/21. At exactly four disagreement labels, the power is 1/14.

## Resource accounting

D and the disagreement indices must be known for the entire declared pool. Both models therefore need to produce predictions on those frames: a completed shadow deployment can supply them, or a logged audit must explicitly pay for backend calls, bytes, and preprocessing. The analytic budget counts manual labels after obtaining those paired predictions. If only a subset was offloaded, the result applies to that selected pool. Gateway latency, radio energy, or counterfactual results on never-shadowed frames do not follow from the certificate.

## Independent verification

```bash
python3 -m unittest discover -s tests -p test_certificate.py -v
python3 scripts/verify_certificate.py --compare-primary --json results/certificate_audit.json
```

The reference verifier uses integer binomial coefficients and rational probabilities. It first enumerates every subset of a canonical finite population for every possible N≤10, D, R, label budget and choice of design. All labeled placements of agreements, corrections and regressions are independently enumerated for N≤6. The optional final comparison invokes the numerical production implementation. The coverage calculation and the power calculations precede that import.
