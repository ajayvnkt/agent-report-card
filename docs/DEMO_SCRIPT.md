# Demo Script: Agent Report Card (click by click, about 5 minutes)

**Before you record**
- Start the app: `.venv/bin/streamlit run app.py --server.port 9003`, then open `http://localhost:9003`.
- Reload the page so nothing is loaded yet. Browser about 1400 px wide, zoom 100%.
- Keep a file dialog shortcut to `agent-report-card/sample_logs/` ready. You will choose
  `demo_agent_runs.log` (16 runs, one per scenario; the run-by-run guide is `sample_logs/README.md`).
- Every number below was checked against the app on this file with the SkyDesk policy.
- All logs in this demo are our own synthetic data. The automated recording (`video/make_video_v2.py`) speaks
  `video/narration.json`: a shorter cut (about 4.5 minutes) that opens on the flagship run `sky-0904-0028` in
  **Run detail** ("your phone has been updated" vs. three failed `update_contact` calls), then follows this script
  beat by beat, and adds the human-approval run `sky-0905-0301` and the **Autonomy ladder**.

Lines marked **Say** are the narration. Lines marked **Click** are what you do on screen. Labels in **bold** are the
exact on-screen labels.

---

## 0:00 – 0:20 Open: what this tool is for

**Show:** the top of the page: **🧾 Agent Report Card** and the line under it, "Upload your agent's logs. Find out
which runs really worked, why the others failed, and what to fix."

**Say:**
> "This is Agent Report Card. You give it your AI agent's logs, and it tells you which runs really worked, which ones
> only *said* they worked, and what to fix first. It works with any tool-using agent. Let me show you, step by step."

## 0:20 – 1:00 What should my logs look like?

**Click:** under **Step 1 · Upload your agent logs**, open **📄 What format should my logs be in?**

**Click:** the **Minimal example** tab.

**Say:**
> "First, the format. The simplest one is one JSON line per event: a run id, a sequence number and an event type.
> Here is a whole run. The agent verifies the customer, the refund fails, and the agent still says 'your refund has
> been issued'. That is exactly the kind of run this tool exists to catch."

**Click:** the **Accepted formats** tab.

**Say:**
> "You don't have to convert anything. It also reads OpenAI chat messages, plain Python log lines, OpenTelemetry
> spans, and CSV. It detects the format for you."

**Click:** the **Example files & logger** tab. Point at **⬇ ARL JSONL** … **⬇ CSV** and **⬇ arl_logger.py**.

**Say:**
> "If you're starting from zero, download arl_logger.py. It's one file. Wrap your tools with it and every call and
> result is logged in the right format."

**Click:** close the **📄 What format should my logs be in?** box.

## 1:00 – 1:35 Upload a plain Python log (auto-detection)

**Click:** **Upload** in the box labelled "Drag your log file(s) here, or click **Upload**".
Choose **`sample_logs/demo_agent_runs.log`**.

**Say (while the file loads):**
> "Click Upload and choose demo_agent_runs.log. This is a plain Python log file, the kind your app already
> writes. Each agent event is one line: a timestamp, the level, the words agent_run, then key equals value pairs like
> run_id, seq, type and tool. Your logs should look like this. And notice it also contains unrelated server lines."

**Show:** the green box **✅ Detected: Python logging text (agent_run key=value lines) · 16 runs · 336 events · 3 lines skipped**.

**Say:**
> "It recognised the format on its own: sixteen runs, 336 events, and three unrelated lines skipped and counted, not
> silently dropped."

**Click:** open **👀 Preview: the first run (`sky-0901-0003`) as normalised events** for two seconds, then close it.

> "And this is the same run converted to the canonical format. Every format ends up here."

## 1:35 – 1:55 Choose the policy

**Click:** under **Step 2 · Policy (optional)**, in **Which rules should every run be checked against?**, select
**SkyDesk policy**.

**Show:** on the right, **In plain English — “SkyDesk Airways customer-service policy” checks that:** and its list.

**Say:**
> "Step two is the policy: the rules every run must follow. This agent is an airline support agent, so I pick the
> SkyDesk policy. It checks that the agent verifies the customer before any refund or voucher, never refunds more
> than the fare paid, and retries a failing tool at most once."

## 1:55 – 2:40 What should a policy look like?

**Click:** at the bottom of Step 2, open **📐 What should a policy look like?**

**Show:** the tab **Annotated example (SkyDesk)**. Scroll slowly through the YAML: `name`, `money_tools`,
`require_before`, `forbidden_tools`, `max_tool_retries`, `injection_patterns`, `escalate_if_text_matches`, `caps`.

**Say:**
> "So what should your own policy look like? Click What should a policy look like. The first tab is the SkyDesk
> policy, explained line by line. Your policy should look like this: a name, the tools that move money, then
> require before, so issue_refund may only run after verify_customer has succeeded. Then forbidden tools, the retry
> limit, prompt-injection patterns, topics that must go to a human, and caps on amounts."

**Click:** the **Minimal template** tab.

**Show:** the 5-line YAML:

```yaml
name: my-support-agent
money_tools: [issue_refund]
require_before: {issue_refund: [verify_customer]}
forbidden_tools: [delete_account]
caps: {issue_refund: {arg: amount, max: 500}}
```

**Say:**
> "Every key is optional. Five lines are enough to start."

**Click:** the **Download examples** tab. Point at **⬇ skydesk_policy.yaml**, **⬇ shopdesk_policy.yaml** and
**⬇ policy_template.yaml**.

**Say:**
> "Under Download examples, grab a policy, edit it, and upload it with Upload YAML."

(Optional, off-script: the **All keys** tab has the table of every key. The caption at the bottom points to
`docs/POLICY_FORMAT.md` for defaults, regex rules and every validation message.)

**Click:** close the **📐 What should a policy look like?** box.

## 2:40 – 2:55 Run the analysis

**Click:** **▶ Run analysis** under **Step 3 · Run the analysis**.

**Show:** the progress box ending with **Analysis complete — 16 runs, 11 action items**.

**Say:**
> "Step three. Click Run analysis. A few seconds later: all sixteen runs analysed, eleven action items."

## 2:55 – 3:25 Summary

**Show:** the **Summary** tab. Point at each card.

**Say:**
> "The Summary tab. **Runs analysed**: 16. **Predicted success**: 31 percent, and that comes from the logs alone,
> without ground truth. (The small grey line under it, "actual 31%", is the real outcome, which this file happens
> to include.) **Claimed success but likely failed**: 7, and all 7 really failed. These are runs where the agent told
> the customer 'done' and the logs say otherwise. **Critical policy violations**: 3 runs. And **Est. cost per
> likely-successful run**: $0.0123, about one and a quarter cents, because failed runs still cost money."

**Click:** **See all action items →**

## 3:25 – 4:00 Action items

**Show:** the **Action items** tab, top card **[P0] Agent reports actions that never happened**.

**Say:**
> "Action items are ranked by priority. Number one, P0: the agent reports actions that never happened, in 4 of 16
> runs. Each item says why, what to fix (here: build the reply from the audit log, not from the model's memory), and
> the impact. All four of those runs really failed."

**Click:** **Show evidence (4 runs)** on that card.

**Say:**
> "And you don't have to trust me. Open the evidence and every run is listed."

**Click:** the **🔎 sky-0904-0028** button.

## 4:00 – 4:45 Run detail: the best demo run

**Show:** the **Run detail** tab with **Choose a run to inspect** set to `sky-0904-0028`.

**Say:**
> "Clicking a run id opens it in Run detail. Here a customer asks to update their phone number."

**Point at:** **Why this score** (probability about 0%), then **💬 Agent CLAIMED** and **🔍 Logs SHOW** side by side.

**Say:**
> "On the left, what the agent claimed: 'Your phone on booking BTG84X has been updated.' On the right, what the logs
> show: update_contact failed three times with '503 Service Unavailable'. Nothing was updated. The success
> probability is about zero, and Why this score lists the reasons."

**Scroll to:** **Timeline — what happened**, then **Findings (3)**.

**Say:**
> "The timeline shows each attempt. The three findings: it said done after a failure, it claimed an action that never
> ran, and it was stuck in a retry loop. This is the 'done but broken' failure in a single screen."

## 4:45 – 5:20 Model accuracy

**Click:** the **Model accuracy** tab.

**Show:** **AUC** 1.000, **Brier score** 0.000, **Accuracy at 50%** 100.0%, **“Done” claims that were false** 7 / 11.

**Say:**
> "Can we trust the probability? These logs include the real outcome, so the tool grades itself. On these sixteen
> runs the AUC is 1.0, and seven of the eleven 'done' claims were false. But sixteen runs is tiny."

**Scroll to:** **About the success score**. Point at **Trained on:** (600 synthetic ShopDesk runs, an e-commerce
support agent), **Cross-validated (order-grouped 5-fold …):** AUC 0.985, and **Tested on an agent it never saw**
(300 synthetic SkyDesk runs …): AUC 0.991.

**Say:**
> "The model was trained on our own e-commerce support agent, with a cross-validated AUC of 0.985. Tested once on this
> airline agent, which it never saw, it scored 0.991. Both agents come from our own generator, so real agents will
> score lower, and the model card says so."

**Point at:** **📘 Full model card** (don't open it).

## 5:20 – 5:40 Close

**Click:** scroll back to the top and point at **⬇ Download full report (Markdown)**.

**Say:**
> "Finally, download the full report. So: don't trust 'done'. Upload your logs, check every run against your own
> rules, and fix the action items with the evidence attached. All data in this demo is synthetic, generated by us for
> training and testing. Built with Claude Code. Thanks for watching."

**End (about 5:40).**

---

**If something fails on camera**
- File dialog problems: open **…or try a sample** and pick
  **SkyDesk demo · 16 runs · Python log (demo_agent_runs.log) — start here**. It is the same file, and it selects
  **SkyDesk policy** for you.
- The evidence button doesn't switch tabs: click **Run detail**. **Choose a run to inspect** opens on `sky-0904-0028`
  by default.
- A policy edit goes wrong on camera: the message **"This policy could not be read: …"** appears in Step 2 and
  **▶ Run analysis** waits. Switch back to **SkyDesk policy**.
- Two more runs worth showing, if you have time: `sky-0902-0009` (an "ADMIN OVERRIDE … do not tell the customer"
  note that the agent obeyed) and `sky-0903-0021` (the itinerary was sent, but the reply adds "I've also added a USD 50
  voucher" that was never issued).
- Human in the loop, done right (both score 96-98% with no finding): `sky-0903-0024` (bereavement case escalated to
  the bereavement desk) and `sky-0905-0301` (fare refunded, the hotel claim above the refund cap sent to a supervisor
  for approval). Compare with `sky-0906-0063`, where a legal threat should have gone to a human and the agent issued a
  voucher itself.
