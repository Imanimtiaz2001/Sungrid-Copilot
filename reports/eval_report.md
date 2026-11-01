# Evaluation report

Embedder: `hashing:4096` · LLM: `none (extractive)` · top_k=4 · 92 golden cases (data/eval/golden_set.jsonl)

## Headline metrics

| Metric | Value | CI gate |
|---|---|---|
| Case pass rate (all assertions) | 92.4% |  |
| Classification accuracy (label in accepted set) | 97.1% | ≥ 85.0% |
| Classification accuracy (strict primary label) | 95.6% |  |
| Intent / routing accuracy | 100.0% | ≥ 95.0% |
| Tool routing precision / recall | 100.0% / 100.0% |  |
| Eligibility decision accuracy | 100.0% | = 100% |
| **Hard-gate violations** | **0** | = 0 |
| Retrieval hit@k (expected doc retrieved) | 100.0% | ≥ 90.0% |
| Answer fact recall (expected fact in answer) | 91.4% |  |
| Off-topic abstention recall | 90.0% |  |
| False abstentions on in-domain questions | 0 |  |
| Unfiltered-fallback rate (knowledge turns) | 3.4% |  |
| Filter widened to top-2 categories | 5.1% |  |
| Latency p50 / p95 per case | 10 ms / 17 ms |  |

Quality gates: classification_accuracy ✅, intent_accuracy ✅, retrieval_hit_at_k ✅, eligibility_decision_accuracy ✅, hard_gate_violations ✅

## Retrieval filter ablation (knowledge questions)

| Strategy | hit@k | MRR |
|---|---|---|
| no_filter | 100.0% | 0.983 |
| hard_filter_predicted | 98.3% | 0.966 |
| system_policy (soft filter + fallback) | 100.0% | 0.983 |
| oracle_filter | 100.0% | 0.983 |

## Classification confusion (rows = expected primary, cols = predicted)

| expected \ predicted | billing_account | company_updates | incentive_rebate | non_relevant | program_policies | technical_installation |
|---|---|---|---|---|---|---|
| billing_account | 10 |  |  |  |  |  |
| company_updates |  | 7 |  |  | 1 |  |
| incentive_rebate |  |  | 19 |  |  |  |
| non_relevant |  |  |  | 9 |  | 1 |
| program_policies |  |  |  | 1 | 10 |  |
| technical_installation |  |  |  |  |  | 10 |

## Document category audit (seed-only classifier over each document's chunks)

| Document | Manifest categories | Classifier top-2 (share) | Entropy (bits) |
|---|---|---|---|
| 01_program_policies | program_policies | program_policies 0.61, incentive_rebate 0.22 | 1.61 |
| 02_incentive_rebate_programs | incentive_rebate | incentive_rebate 0.87, program_policies 0.04 | 0.8 |
| 03_billing_faqs | billing_account | billing_account 0.71, incentive_rebate 0.15 | 1.34 |
| 04_technical_installation_guidance | technical_installation | technical_installation 0.85, program_policies 0.05 | 0.89 |
| 05_company_updates | company_updates | company_updates 0.78, program_policies 0.13 | 1.12 |
| 06_ambiguous_rebate_billing_adjustments ⚠️ multi-label | incentive_rebate, billing_account | incentive_rebate 0.52, billing_account 0.38 | 1.51 |
| 07_installer_certification | program_policies | program_policies 0.86, technical_installation 0.05 | 0.88 |
| 08_governance_voting | program_policies | program_policies 0.90, company_updates 0.04 | 0.65 |
| 09_battery_storage_incentive | incentive_rebate | incentive_rebate 0.85, technical_installation 0.07 | 0.87 |
| 10_low_income_bill_credit | incentive_rebate | incentive_rebate 0.73, billing_account 0.16 | 1.25 |
| 11_net_metering_trueup | billing_account | billing_account 0.89, company_updates 0.04 | 0.72 |
| 12_autopay_failure_policy | billing_account | billing_account 0.87, program_policies 0.05 | 0.82 |
| 13_battery_storage_installation | technical_installation | technical_installation 0.67, incentive_rebate 0.16 | 1.49 |
| 14_annual_impact_report | company_updates | company_updates 0.74, incentive_rebate 0.14 | 1.28 |

## Failed cases

- `pp-02`: category non_relevant not in ['program_policies']
- `ir-02`: answer lacks any of ['30 days']
- `ir-04`: answer lacks any of ['income attestation', 'interconnection application']
- `ba-01`: answer lacks any of ['21 days']
- `ba-06`: answer lacks any of ['Account Settings']
- `amb-03`: answer lacks any of ['acknowledge']
- `nr-09`: category technical_installation not in ['non_relevant']; route knowledge != abstain
