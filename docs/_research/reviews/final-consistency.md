# Final consistency review — after the fix pass

**Date:** 2026-10-24. **Scope:** `README.md`, `docs/00`–`docs/07`, `docs/SOURCES.md`, read against the two prior audits (`source-audit.md`, `harness-audit.md`, `strategy-audit.md`) and the local sources (`harness_readme_raw.md`, `getting_started_notebook.txt`). **Question:** did the fixes introduce new contradictions, and do any Critical/High issues survive? Audit only; no doc was edited.

**Method:** every item on the eight-point checklist was grepped corpus-wide (not spot-checked), then each hit was read in context and compared with the other docs and the ledger entry it cites. New text introduced by the fix pass (ledger S25/S30/S36/S37/S39/S50 additions, `02` §2.1 "Timeout correction", `01` §8.3, `03` §3/§7.1/§7.2/§9.4, `04` §1.2/§2.1/§3.2b/§5.1/§9.2) was checked for internal agreement and against the raw sources.

**Verdict:** 4 of 8 checklist items pass cleanly; 4 fail (Q1, Q4, Q5, Q8). Seven High findings in total — F6 is an internal contradiction outside the eight questions. No new Critical defect was introduced: the 240 s/300 s mechanism error, the per-task score step, the dates, the chat template, the wheelhouse install, the tuple bug, and the `0.24` qualification are all genuinely fixed.

---

## 1. Checklist verdicts

| # | Question | Verdict | Finding |
|---|---|---|---|
| 1 | Primary budget unambiguous (5/24/2048 vs 8/28/4096) | **No** — 8/28/4096 is still the "starting point" in 5 places, and the runbook measures the wrong bundle | F1 |
| 2 | `0.24` consistently author-reported, never "verified" | **Yes** — every notebook-derived instance carries *claims*/author-reported; leaderboard `0.24` is separately sourced (S4) and is a different fact | — |
| 3 | Phase-2 timeout 300 s fixed, S38's 240 s flagged | **Yes** — `01` §2.2/§8.1, `02` §2.1 "Timeout correction" and §2.3, `04` §2.1 all scope 240 s to Phase 1; the S38 ledger entry itself records the contradiction | — |
| 4 | Alpha/rank consistent (rank 16, alpha 8/16, 32 diagnostic) | **No** — `04` §8.3 sweeps 8/16/32 with no diagnostic label; rank 8 is offered as a start without the alpha 4/8 guard | F2 |
| 5 | Training base IDs consistent, verified vs `[VERIFY]` | **No** — CT repo ID casing differs between docs; one unverified training ID is a literal manifest value | F3, F4 |
| 6 | Per-task score steps (1/58, 1/60, 1/120) | **Yes** — `00` §3 and `06` §9 agree exactly; no `0.0083`-as-final-step instance survives | — |
| 7 | Dates (entry Nov 25, paper Nov 12, final Dec 2, freeze Nov 28) | **Yes** — `00` §2, `README` §5, `03` §12, `07` §1.6 all agree; the Nov 24 entry date is explicitly "one day margin" | — |
| 8 | Remaining unsupported quantitative claim or S-number mismatch | **Two mismatches** — `05` R1 and `03` §7.1 cite entries that do not contain the claim | F5 |

Plus one internal numeric contradiction on trajectory generation (F6) and one measurement-command contradiction inside the budget item (F1e).

---

## 2. Findings (Critical/High only)

### F1 — High — 8 min/28 calls/4096 is still presented as the starting point, and the baseline is measured at the wrong budget

The corpus now says "primary 5 min/24 calls/thinking 2048" in `README` §4, `02` §2.1, `04` §2.1, `02` §10. These five sites still send the reader to the reference config:

| Site | Text as written | Exact fix |
|---|---|---|
| `02` §3.3 (line 198) | "4096: public prompt-only baseline; more deliberate, slower (S36)." | "4096: reference config only (S36); not the primary bundle." |
| `02` §3.3 (line 201) | "Start at 4096 for the public baseline." | "Start at 2048 for the primary bundle; 4096 is the A/B arm." |
| `02` §10 (line 432) | "`max_output_tokens` 8192, thinking 4096, temp 0.2, top_p 0.95." — immediately above line 433's "Primary … thinking 2048" | Delete line 432; fold into line 433: "`max_output_tokens` 8192, temp 0.2, top_p 0.95, thinking 2048 (primary) / 4096 (A/B)." |
| `01` §3.4 (line 163) | "…`thinking_budget: 4096`… (S36). **This is a reasonable starting point.**" | "…This is the reference config; the primary bundle uses `thinking_budget: 2048` (see `02` §2.1)." |
| `02` §4.4 (line 251) | Prompt skeleton: "- 28 tool calls and 8 minutes." | "- 24 tool calls and 5 minutes." (the skeleton is the template a reader ships; `04` §2.1 already instructs this edit) |
| `README` §5 W2 (line 115) | "5-min/24-call vs 6-min/28-call paired test" | "5-min/24-call vs 8-min/28-call (the reference bundle)". There is no 6-min/28-call bundle anywhere else, and `README` §9 calls 8/28 the fallback. |

**F1e — the runbook measures the reference budget, not the primary.** `04` §3.3 (lines 337–338) and §5 (lines 413–414) pass `--max-tool-calls 28 --max-time-minutes 8` on the CLI, which either overrides the `eval_config.yaml` the reader was just told to set to 24/5 (so `agent_p50`/`setup_p50` and the 12 h projection describe the 8-min bundle) or is redundant. The docs never say which wins.

**Fix:** drop both flags from §3.3 and §5 (let `eval_config.yaml` decide), or set `--max-tool-calls 24 --max-time-minutes 5`, and add one line to `01` §10: "CLI budget flags override `eval_config.yaml`."

---

### F2 — High — Alpha/rank instructions still conflict in two places

| Site | Text as written | Exact fix |
|---|---|---|
| `04` §8.3 (line 608) | "Edit `adapter_config.json` `lora_alpha` to 8/16/32; restart" | "Pin rank 16; edit `lora_alpha` to 8/16 (scale 0.5/1.0). `lora_alpha: 32` (scale 2.0) is a **diagnostic only** — the documented degenerate point at rank 16 (S25). If you use rank 8, sweep alpha 4/8." (mirror `03` §9.4 verbatim) |
| `03` §1 (line 17) + §5.2 (line 224) | "Adapter rank: 8–16 start" / "Rank 8–16 is the conservative start." while line 18 sweeps alpha 8/16 | Append to both: "at rank 8 sweep alpha 4/8 (scale 0.5/1.0); the 8/16 sweep assumes rank 16." |
| `README` §9 (line 176) | "train a rank-8–16 LoRA" | "train a rank-16 LoRA" (matches `README` §4 Path B and `03` §7.1/§7.2) |

The rule itself is now correct in `README` §4/§7, `03` §9.4, `05` R6, `06` T3; these three sites are the residue that lets a reader pair rank 8 with alpha 16 (= scale 2.0).

---

### F3 — High — CT base repo ID is written with two different casings; one of them cannot be the HF repo

- `03` §3 "Practical rule" (line 59): "Serve the adapter on `google/gemma-4-31b-it-qat-w4a16-ct`."
- `03` §9.1 (line 580): "model: `google/gemma-4-31b-it-qat-w4a16-ct` (local path)".
- `04` §0.4 (line 91) and the S8 URL use `google/gemma-4-31B-it-qat-w4a16-ct`.

Hugging Face repo IDs are case-sensitive; `google/gemma-4-31b-…` is not the repo S8 documents, so a copy-paste `hf download`/`vllm serve` fails or resolves to nothing. The lowercase form is the **competition model alias** (S1/S5), not an HF ID.

**Fix:** use `google/gemma-4-31B-it-qat-w4a16-ct` for every HF repo reference (03:59, 03:580), and add one clause where the alias first appears (`00` §6): "declare the bare alias `gemma-4-31b-it-qat-w4a16-ct` in `agent.yaml`; the HF repo is `google/gemma-4-31B-it-qat-w4a16-ct` (S8)."

---

### F4 — High — An unverified training-base ID is a literal manifest value

`06` §4 (line 190) records `"training_base": "google/gemma-4-31B-it-qat-q4_0-unquantized"` with no marker, while `03` §3 (line 59) says to "verify its exact repo ID and revision before download" and `03` §7.4 marks the 12B/E4B equivalents `[VERIFY exact repo ID/revision]`. No ledger entry verifies an org/prefix for any unquantized-QAT repo — S7/S43 give only the suffix convention.

**Fix:** `"training_base": "google/gemma-4-31B-it-qat-q4_0-unquantized  # [VERIFY] repo ID + revision before download; fallback google/gemma-4-31B (S12)"`, or set the value to the verified `google/gemma-4-31B`.

Same class, one line: `03` §7.3 (line 474) `model_name="unsloth/gemma-4-31B-it"` — append `[VERIFY]`; S10 records the API surface, not that repo ID.

Verified and consistent, leave alone: `03` §3/§7.1/§7.2 `google/gemma-4-31B` (S12).

---

### F5 — High — Two citations still point at entries that do not contain the claim

| Site | Text as written | Problem | Exact fix |
|---|---|---|---|
| `05` R1 (line 15) | "The public 0.18 notebook warns that its 8-minute per-task cap is not a guarantee of global completion (S38)." | S37 owns that sentence; the S38 entry says the opposite ("caps kept a scored run under 12h"). `01` §8.3 already cites S37 for it, so `05` and `01` disagree about who warned. | `(S37)`; keep "0.18/0.22" if you want both notebooks. |
| `03` §7.1 note (line 411) | "Axolotl's official example uses `gemma4` (S12)" | The S12 ledger entry records `base_model`, `load_in_4bit`, `adapter`, `lora_r/alpha`, target regex, `sequence_len`, packing, batch, LR, `max_grad_norm`, `bf16`, `tf32`, `gradient_checkpointing`, `activation_offloading`, `gemma4_hybrid_attn_impl`, `attn_implementation` — no `chat_template`. | Add `chat_template: gemma4` to S12 with its value, or drop the clause and write "if your Axolotl version rejects `gemma-4-thinking`, …". |

These are the only two survivors of the audit's S-number list; the S28/S30/S36/S39/S25/S11/S10/S7 mismatches were genuinely repaired by extending the ledger, not by weakening the prose.

---

### F6 — High — Trajectory generation is "five temperatures" in prose and four in the loop

- `04` §6.1 (line 462): "[ESTIMATE] 129 tasks × 5 temperatures ≈ 645 runs; … ~65 h sequential."
- `04` §6.1 (line 464): "For each of five temperatures between 0.2 and 1.0 (e.g. 0.2, 0.4, 0.6, 0.8, 1.0)…"
- `04` §6.1 (line 467): `for T in 0.2 0.5 0.8 1.0; do` — four, and not the same four values.
- `03` §4.1 (line 82): "at five temperatures between 0.2 and 1.0 (e.g. 0.2, 0.4, 0.6, 0.8, 1.0)".

The command a reader actually runs produces 516 runs (~52 h), not the 645/65 h the week is planned around, and the two temperature grids differ.

**Fix:** pick one grid and use it in all four places — e.g. `for T in 0.2 0.4 0.6 0.8 1.0` and keep 645/65 h; or keep four and change both estimates to "129 × 4 ≈ 516 runs; ~52 h" and `03` §4.1's example list to `0.2, 0.5, 0.8, 1.0`.

---

### F7 — High — The promotion rule is still "≥2 tasks" in four places, against four places that say ≥8 discordant/≥75%

Consistent (correct) sites: `README` §4 (line 90), `02` §8.7 (line 405), `03` §9.5 (line 636), `05` §5 decision tree (line 401).

Contradicting sites:

| Site | Text as written | Exact fix |
|---|---|---|
| `06` §3.4 (lines 130–135) | "It wins ≥2 more tasks … A 2-task win on 40 tasks is weak but worth a submission" | Replace item 1 with "It wins ≥8 discordant pairs with ≥75% wins (or two seeds)", and the closing sentence with "A 2-task net win at 20–40 tasks is inside the sign-test noise (d=8, 6–2 → p≈0.29); do not spend a slot on it." |
| `04` §8.4 (line 620) | "Promote only if the adapter wins by ≥2 tasks and does not regress time/patch rate." | "Promote only if the adapter wins ≥8 discordant pairs with ≥75% wins (or two seeds) and does not regress p95/patch rate." |
| `03` §1 (line 23) | "Success criterion: ≥2-task paired local win …" | "≥8 discordant pairs and ≥75% wins (or two seeds) plus a hidden score" |
| `03` §13 (line 728) | "Paired local eval ≥2-task win" | "Paired local eval ≥8 discordant/≥75% wins" |

`06` is the document that owns decision rules, so it currently contradicts the two docs that consume it.

---

## 3. Checked and clean (no action)

- **0.24:** all six notebook-derived instances say *claims* / author-reported (`README` 5, `00` §7, `02` 5, `03` 3, `04` 8, `05` R12); leaderboard 0.24 is separately S4-sourced; `README` 81 adds "0.24 is not guaranteed". No unqualified instance.
- **300 s:** `01` §2.2, §8.1 table, `02` §2.1/§2.3, `04` §2.1. The 240 s value is consistently scoped to Phase 1; the false "shorter timeout fails a correct patch" rationale is gone.
- **Score steps:** `00` §3 and `06` §9 both give 1/58 ≈ 0.017 public, 1/60 ≈ 0.017 final, 1/120 ≈ 0.0083 blended; `00` §9 bands and `06` §9 truncation note agree.
- **Dates:** entry/merger 2026-11-25 (`00` §2/§8, `README` §10), with `README` §5 W5's "Nov 24 (official Nov 25; one day margin)" now explicit; paper 2026-11-12 (`README` §4 Path D, `07` §1.1/§1.6); freeze Nov 28 (`README` §5, `03` §12); final 2026-12-02 with Dec 2 as buffer only.
- **Verified line refs:** S5:153 (LoRA right-sizing), S5:506 (300 s fixed), S5:130/326 (skill extensions/tools), S6:341/344 (eval keys), S6:282-284 (computed LoRA params), S6:352 (2-tuple) all match the raw files; the `01` §1, `01` §5 step-16, `04` §1.2, §2.2, §3.2b, §9.2 fixes are correct.
- **Arithmetic:** prize sums ($65k+$35k+$10k), 120×{5,5.5,8} = 10/11/16 h, 245 MB rank-16 adapter, 3 GiB = 3,221,225,472 B, sign-test p-values in `06` §3.3.

---

## 4. Below the Critical/High bar (listed so they are not lost)

- `03` §9.5 (line 636) "…not enough (F13)" and `04` §6.4 (line 516) "…drop the adapter (F17)" — dangling audit IDs; the corpus's evidence convention is `[S#]`/`[ESTIMATE]`/`[COMMUNITY]`. Delete both parentheticals.
- `02` §5.2 (line 306) "continue from `end_line`" — `read_file` is inclusive, so this re-reads one line; should be `end_line + 1`.
- `02` §4.2 (line 228) "A 3,700-character prompt is ~900–1,000 tokens" — uncited (S36 records no length). Label `[ESTIMATE]` or say "a 3–4k-character prompt".
- `06` §2.2 splits sum to 130 tasks at the top of every range (60+30+30+10) against 129 public tasks; `04` §4.1 asks for 10–15 calibration tasks while `06` §2.2 says 5–10.
- `03` §4.2 (line 99) and `05` R7 (line 155) call `gemma-4` the "non-thinking variant"; S10 describes a size split (thinking template for 31B/26B, `gemma-4` for small models).
- `README` §1.3 points at "`docs/02-agent-engineering.md` §6" for the public config; it is §2.
