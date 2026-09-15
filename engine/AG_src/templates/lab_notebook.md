# Lab Notebook - {run_id}
## Iteration {iteration}
### Date: {date}

---

### Hypothesis
{hypothesis}

---

### Parameter Changes from Previous Iteration
{parameter_changes_table}

---

### Pipeline Configuration
{config_summary}

---

### Results Summary

#### Step 04 - ESMFold QC
- Total candidates: {n_total}
- Passed pLDDT gate (>={plddt_gate}): {n_passed}

#### Step 05 - Docking
- Engine: {dock_engine}
- Top {dock_top_pct}% candidates: {n_docking_passed}

#### Step 06 - Rosetta Refinement
- Candidates refined: {n_refined}
- Passed ddG gate (<={ddg_gate}): {n_ddg_passed}

#### Step 07 - Analysis
- FoldMason lDDT range: {lddt_min}-{lddt_max}

---

### Top Candidates
{rank_table_top10}

---

### Critic Analysis
{critic_notes}

---

### Next Iteration Plan
{next_plan}
