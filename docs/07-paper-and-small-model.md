# 07 — Paper Track and Small-Model Bonus

These are the two hedges that do not require beating the main 31B leaderboard. Both have lower competition density than the main track, but neither is free: the paper requires a real research contribution, and the small-model bonus requires a working E2B/E4B agent plus one of your two final submission slots.

---

## 1. Paper track

### 1.1 Facts

- Separate Kaggle competition (S46).
- Total prizes: **$35,000**:
  - Overall best paper: **$15,000**.
  - Best new resource (dataset, library, tool): **$10,000**.
  - Best new application: **$10,000**.
- Deadline: **2026-11-12, 11:59 PM UTC** (S46).
- Scope: **3,000 words maximum**, original, unpublished research (S46).
- Non-archival: submitting does not prevent submitting to other venues (S46).
- Submission: a Kaggle Writeup with title/subtitle, abstract, introduction, methods/experiments, related works/citations. Optional public notebook and/or arXiv-ready PDF (S46).
- Judging: novelty, quality, relevance, verifiability, clarity; each 0–5; final score is the average (S46).
- At retrieval: 5,620 entrants, 338 participants, 327 teams, 348 submissions (S46).

### 1.2 What the judges are asking for

| Criterion | What it means | What to show |
|---|---|---|
| Novelty | New insights, deeper understanding, important properties | A clear claim that is not obvious from the public notebooks |
| Quality | General and universal beyond the competition | Method that transfers to other SWE agents/models |
| Relevance | Impact on software engineering/agentic learning | Why the result matters beyond the leaderboard |
| Verifiability | Enough information to understand and reproduce | Data, code, prompts, configs, hashes, negative results |
| Clarity | Clearly presented and written | Tight structure, figures/tables, no hype |

The paper track is not a “leaderboard diary.” A case study can win if it is rigorous and generalizable. A new dataset/tool/application can win the resource/application prizes even if the main score is modest.

### 1.3 Strong paper angles from this competition

These are grounded in public evidence and your own experiments:

1. **Evaluator drift and score noise.** The scorer changed repeatedly (swegemma 0.2.7 → 0.2.11, adk-submission 0.2.11 → 0.2.13, vLLM rebuilds) and identical packages scored differently (S39). A rigorous case study on how to compare agents when the evaluator moves is novel and useful. The public Part 2 blog is a model (S39).
2. **Trajectory quality vs quantity for LoRA.** Public evidence: 697 per-call samples did not beat the base; 1,500 trajectories were generated; quality filtering can make a 10% subset outperform the full set (S19, S25). A controlled study on SWE-agent trajectories, with step/segment masking, is relevant.
3. **Budget-aware agent design.** The public prompt-only config’s (author-reported 0.24) edit-by-call-12 rule and 28-call/8-min budget is a concrete intervention (S36–S38). A study of “when to stop exploring and edit” across models/tasks is generalizable.
4. **Local-to-hidden transfer.** Public teams’ local screens did not predict hidden scores (S35, S39). A methodological paper on repository-grouped holdouts, paired comparisons, and noise floors is valuable.
5. **Graph tools for code navigation.** The competition provides AST graphs/embeddings; public audits found them weak (S36–S38), and async functions are missing (S33). A careful study of when graph tools help (symbol resolution, callers, cross-file dependencies) is a “code comprehension” contribution.
6. **Small-model distillation.** E4B/E2B with teacher trajectories from 31B/stronger models; the small-model bonus provides a natural evaluation. A resource paper with a public trajectory dataset and a reproducible training recipe fits the “new resource” prize.
7. **Test-time verification.** Agents can’t see hidden tests; how much self-verification helps vs costs time? A controlled study of verification policies (one test, two tests, inline assertions) is relevant.

Avoid:

- “We got 0.24 with prompt X” without a generalizable insight.
- A pure leaderboard report with no controls.
- Claims that local gains transfer without paired evidence.
- Hiding negative results; the paper track values verifiability.

### 1.4 Suggested paper outline (3,000 words)

1. **Title/subtitle** — specific claim, not “Gemma 4 Agent.”
2. **Abstract** — problem, method, key result, implication (150–200 words).
3. **Introduction** — why offline SWE agents matter; what is missing; your contribution (400–500 words).
4. **Related work** — SWE-bench/SWE-agent, LoRA/QLoRA for code agents, trajectory distillation, budgeted agents (400–500 words).
5. **Method** — agent scaffold, data generation, filters, training, evaluation; enough to reproduce (700–900 words).
6. **Experiments** — datasets/splits, metrics, paired design, results, ablations, negative results (700–900 words).
7. **Discussion/limitations** — noise, evaluator drift, transfer, compute (300–400 words).
8. **Conclusion** — one paragraph.
9. **References** — cited works.
10. **Appendix/links** — code, data, prompts, configs, hashes (not counted in the 3,000 words if linked).

Figures/tables:

- A pipeline diagram.
- A paired comparison table with discordant counts.
- A budget/time vs resolve plot.
- A data-quality ablation.
- A failure taxonomy.

### 1.5 Verifiability package

Prepare:

- Public code repository (Apache 2.0 or compatible).
- Public trajectory dataset (if allowed by licenses/rules).
- Exact prompts/configs/eval settings.
- Model/adapter revisions and hashes.
- Run manifests and per-task results.
- A README that reproduces the main table with commands.
- A limitations section that states what did not work.

Winners must provide training code and methodology (S3). Build the package as you go; don’t reconstruct it after winning.

### 1.6 Timeline

- **Now–Nov 1:** choose angle; collect data/results; write outline.
- **Nov 1–7:** draft method/experiments; make figures; run missing controls.
- **Nov 7–10:** internal review for novelty/verifiability/clarity; check citations.
- **Nov 10–11:** finalize 3,000 words; publish code/data; submit.
- **Nov 12:** deadline.

Do not wait for the main leaderboard result. A paper on a well-controlled negative result can be stronger than a thin “we placed” report.

---

## 2. Small-model bonus

### 2.1 Facts

- **$10,000** for the highest-ranking submission using Gemma **E2B or E4B** (S47).
- Must explicitly use E2B/E4B as the base model (S47).
- Must be one of the team’s **two selected final submissions** (S47).
- LoRA fine-tuning is allowed; same submission rules/tie-breaking as the main track (S47).
- Cross-track eligible: small-model entries remain eligible for main leaderboard awards and the Paper Track (S47).
- A separate daily notebook displays filtered small-model standings (S47).
- [COMMUNITY] Best small-model public score ~**0.06** as of 2026-10-20 (S47; one snapshot, not a measured floor).

### 2.2 Why it is attractive

- The main track top is 0.24 with 3,548 teams; the small-model bar is ~0.06 in the community snapshot (S4, S47).
- E4B LoRA trains on ~17 GB (S10), so one A100 40 GB is enough; iteration is cheap.
- The bonus is additive and cross-eligible; a small-model entry can still win main/paper awards.

### 2.3 Why it is risky

- E4B solved **0/31** tasks in one public comparison, while 31B solved 9–11/31 (S34). It submits a patch on ~97% of tasks (S34); the “often edits the wrong file” part is an inference from its 0/31 solve rate, not file-level data [INFERENCE].
- The small-model lane still requires a working agent; a weak scaffold will score 0.
- **Selecting** it consumes one of your two final slots; a probe does not. If your 31B entries are strong, the opportunity cost of selection may exceed $10k expected value.

### 2.4 Recommended approach

1. Reproduce the 31B prompt-only baseline first; don’t start with E4B.
2. Build a separate E4B agent with the same prompt/budget discipline; adjust for its weaker localization:
   - more emphasis on reading the issue’s exact names/messages;
   - earlier edit (call 8–10);
   - fewer tools/calls (E4B is fast but wanders);
   - targeted verification.
3. Train E4B LoRA on E4B or 31B trajectories:
   - E4B LoRA ~17 GB; one A100 40 GB.
   - rank 8–16; alpha = rank; text-only; freeze vision/audio.
   - Use the same thinking mode as inference.
4. Validate on a small-model holdout; compare to the E4B prompt-only baseline.
5. Submit as a final candidate only if it has a real score and does not displace a stronger 31B entry.

### 2.5 Expected-value framing

Let:

- `P_win` = probability your E4B entry is the highest eligible small-model submission.
- `V` = $10,000.
- `C` = opportunity cost of the final slot (expected value of the 31B entry you would otherwise select).

Select E4B only if `P_win × V > C`. With a best score of 0.06 and many teams likely trying, `P_win` may be low unless your E4B agent is clearly better than the public baseline. If your 31B entry has a realistic chance at a main prize, the slot is probably worth more there.

The bonus is most attractive if:

- your 31B post-training line fails;
- you have a strong E4B-specific scaffold/data recipe;
- you have a spare final slot after securing a strong 31B score.

---

## 3. Combining the tracks

A sensible portfolio:

| Slot/asset | Use |
|---|---|
| Final submission 1 | Best validated 31B bundle (prompt-only or adapter) |
| Final submission 2 | Diverse 31B challenger **or** E4B bonus entry |
| Paper track | Independent; submit a rigorous case study/resource |
| Public code/data | Supports paper/winner obligations and community |

Do not put all effort into the paper and neglect the main submission; the paper deadline is earlier (Nov 12), so front-load the writing if you pursue it. Do not put all effort into E4B and leave the 31B baseline unscored.

---

## 4. Paper/small-model checklist

### Paper

- [ ] Angle chosen; novelty claim in one sentence.
- [ ] Related work covers SWE agents, LoRA/QLoRA, trajectory distillation, budgeted agents.
- [ ] Method reproducible (prompts, data, filters, training config, eval).
- [ ] Experiments use paired/held-out design; negative results included.
- [ ] Figures/tables readable; no leaderboard-only claims.
- [ ] Code/data public; licenses compatible; hashes/revisions.
- [ ] 3,000-word limit respected.
- [ ] Submitted before 2026-11-12 23:59 UTC.

### Small model

- [ ] E4B/E2B base explicitly declared; single model rule.
- [ ] Prompt/budget tuned for E4B, not copied blindly from 31B.
- [ ] LoRA (if any) validated on the exact base; A/B output differs.
- [ ] Local paired comparison vs E4B baseline.
- [ ] Hidden score obtained.
- [ ] Final-slot opportunity cost considered.
- [ ] Cross-track eligibility understood.
