# Category decisions and the ambiguous document

## The problem

`06_ambiguous_rebate_billing_adjustments.md` describes what happens when an approved rooftop rebate is reversed after it was paid. The document itself says it "does not sit cleanly" under either category:

- **Incentive & Rebate:** the Incentive Program committee makes the reversal determination, and appeals go through the *Incentive Program appeals process*, not the grievance procedure.
- **Billing & Account:** the Billing team recovers the money over up to 6 billing cycles, autopay handling applies, and manual payers must acknowledge the revised amount.

Members ask about it in both vocabularies: *"my rebate was reversed"* (rebate words) and *"there's a new adjustment on my bill"* (billing words). Whatever the classifier picks, the filter must still reach this document.

## Options considered

| Option | Result | Verdict |
|---|---|---|
| A. Force one label (Incentive) | A billing-worded question ("why is there an adjustment on my bill?") is filtered to billing and **can't reach the answer** | ✗ silent recall loss |
| B. Duplicate the chunks under each category | Works with a filter, but unfiltered search returns duplicates, the index grows, and edits must be kept in sync | ✗ maintenance cost |
| C. A new "cross-cutting" category | The classifier would have to predict "ambiguous", which is not a real topic, and the taxonomy fills with overlap categories | ✗ hurts classification |
| **D. Multi-label documents, single-label queries** | Each chunk stored **once** with `categories: [incentive_rebate, billing_account]`; the filter matches if **any** label matches (Qdrant `MatchAny`) | ✓ **chosen** |

The brief's rule holds: each query gets **one** primary category, used as a filter. Only documents carry more than one label.

## How it is implemented

- `config/kb_manifest.yaml`: `categories: [incentive_rebate, billing_account]`, `primary: incentive_rebate`, `ambiguous: true`, and a written `rationale` (reviewed like code).
- **Primary = incentive_rebate** because the *determination* starts there, and so does the member's next step (the appeal). Billing is the downstream execution. `primary` is used only for display and reporting; it never restricts retrieval.
- Categories are part of each chunk's id, so re-labelling a document re-indexes its payload automatically.

## Evidence that the label is right (not just asserted)

`sungrid audit` classifies every chunk with a classifier trained **only on seed queries**, never on KB text, so the check is not circular, and then aggregates per document:

| Document | Classifier top-2 (share) | Entropy |
|---|---|---|
| 06 (rebate reversal) | incentive_rebate **0.51**, billing_account **0.38** | **1.54 bits** (among the highest) |
| 02 (rebate programs) | incentive_rebate 0.86 | 0.84 |
| 11 (true-up) | billing_account 0.89 | 0.72 |

The classifier sees the same split the manifest records. Single-category documents concentrate on one label; doc 06 divides between exactly its two. A test (`test_audit_flags_the_ambiguous_document`) asserts this. At 10x documents the same audit becomes the triage queue: high-entropy documents go to a human for multi-labelling.

Tests show the effect at runtime: doc 06 is the top hit under **either** filter, and **never** appears under an unrelated filter (`tests/test_retrieval.py`).

## Secondary ambiguities (kept single-label, deliberately)

| Doc | Tension | Decision |
|---|---|---|
| 07 Installer certification | Program rule (next to the installer code of conduct), but a lapsed certification makes the installer *non-approved* for rebates | `program_policies`. Rebate-worded questions still reach it through the low-margin top-2 filter and the unfiltered fallback (measured in the eval report's fallback rate). The eligibility agent also explains "lapsed = non-approved" and cites 07 |
| 10 Low-income bill credit | Shows on the bill, but it is an incentive program (doc 02 lists it among incentive programs) | `incentive_rebate`; the golden set accepts either label for these questions |
| 05 Company updates — multi-family tier | News, but it touches eligibility | `company_updates`. The document itself says not to rely on it for eligibility, and answers quote that warning |

Rule of thumb: **multi-label only when one document is truly owned by two teams** (06). For a single-owner document with a side effect in another area, keep one label and rely on the retrieval safety nets, whose usage is measured.

## What would change this decision

- If the fallback rate for a category rises in production, that is a signal to multi-label or split the documents involved.
- At larger scale, labels should move from document level to **section level**. In doc 06, "Interaction with Autopay" is mostly billing and "Member Notification" (appeals) is mostly incentive. The manifest format extends to sections without code changes to retrieval.
