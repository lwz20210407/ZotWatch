# Literature topic matching and research priorities

The monitoring profile centers on LPBF/TC4 plasticity, constitutive behavior,
fracture and damage, including transferable methods developed for other metals.

## Candidate selection

1. Match hard exclusions against title and abstract.
2. Require at least one configured `required_any_group_sets` rule. Groups within
   a rule are AND conditions; terms within a group are OR conditions.
3. Require an `include_keywords` match when `require_topic_match` is enabled.

Journal names identify sources but no longer count as evidence of a paper's
topic. A paper in Engineering Fracture Mechanics must still match on its own text.

Matching uses word boundaries, Unicode normalization and punctuation/space
normalization. Ordinary trailing plurals are allowed. `DIC` does not match
`prediction`; `gene` does not match `generalized`; `cell` does not match
`cellular`. Short acronyms still require the existing topic groups as context.
Ti-6Al-4V spacing/dash variants share a canonical form. LPBF, L-PBF, SLM,
PBF-LB/M and their configured full process names share another canonical form.
This is lexical matching, not semantic disambiguation or stemming.

The September 2026 revision adds calibration/identifiability, full-field and
post-necking identification, regularization/mesh objectivity, and LPBF
defect/texture/heat-treatment queries. Crystal plasticity, backstress,
brittle-transition and several process/microstructure terms are no longer hard
exclusions. Fatigue, creep, lattice and atomistic studies may enter ranking when
they also satisfy the mechanical topic rules.

## Ranking

The existing similarity, recency and metric score is multiplied by the first
matching `scoring.research_priorities` rule, before recommendation thresholds
are applied. Current coefficients are heuristic and adjustable:

| Research category | Multiplier |
| --- | --- |
| Peripheral reference (fatigue/creep/lattice/atomistic) | 0.70 |
| Transferable methods | 1.00 |
| Mechanism reference | 0.85 |
| TC4 core | 1.00 |
| Other accepted research | 1.00 |

Peripheral matches are checked first; a paper mentioning such a topic can be
demoted even when it also mentions TC4. Transferable calibration methods take
precedence over mechanism references. These coefficients are not calibrated
probabilities and do not guarantee a fixed category order. Downweighted papers
can fall below the existing recommendation threshold; they are not categorically
discarded during topic filtering. The original score, multiplier and category
are retained in each ranked item's `extra` metadata. HTML and RSS show the category.

Run offline regression checks from the project directory:

```powershell
python -B -m unittest discover -s tests -v
```

Tests cover word collisions, aliases, recall across metals, unrelated results,
ranking adjustments and preservation of input candidates. Embedding inference
is stubbed in scoring tests; a successful deployed workflow is a separate
end-to-end check.
