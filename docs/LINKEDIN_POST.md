# LinkedIn post (draft)

---

My AI agent says "done". Is it?

For Week 1 of Mastering Agentic AI at The Gen Academy, I built **Agent Report Card**: upload your agent's logs, click **Run analysis**, and for every run you get what happened, how likely it really worked, and what to fix first.

How it works:
• Upload logs in any of five formats: OpenAI chat messages, plain Python log lines, OpenTelemetry spans, CSV, or its own simple JSONL format. It detects which one by itself.
• Pick or write a policy, a short YAML file with your agent's rules ("no refund before the customer is verified", "never refund more than the fare paid"). The app shows exactly what a policy should look like, with a 5-line template.
• Every run gets a success probability, with no ground truth needed, plus findings: "said done after a tool failed", "claimed an action that never ran", "refund before verification", "obeyed a prompt injection", "retry loop".
• The findings become ranked action items, each with the run ids that prove it.

My favourite example: a customer asks to update their phone number. The tool call fails three times with "503 Service Unavailable". The agent replies: "Your phone has been updated." Predicted success: about 0%.

Honest numbers, on our own synthetic data:
• I generated two different support agents: an online store (ShopDesk, 600 runs) to train the model, and an airline (SkyDesk, 300 runs) to test it.
• Cross-validated AUC on the store agent: 0.985. On the airline agent it never saw: 0.991. In the airline logs, 79 of 220 "done" claims were false.
• Those numbers are optimistic: both agents come from my own generator, with failures I planted. Expect clearly lower accuracy on a real agent, and retrain on your own labelled runs.

How I built it, vibe coding with Claude Code:
v1 was a dashboard for one dataset. The feedback: "this must be a reusable tool: upload logs, run, get a probability per run and action items, for any log format." So I pivoted. I wrote the log format spec first, then ran agents in parallel (format adapters, synthetic data, analyzer and model), integrated, built a guided UI, and let an adversarial QA agent attack it with 25 hostile inputs. It found 9 real bugs that the passing tests had missed. In v3 I made every dataset our own, retrained on one synthetic agent and tested on another, and added a policy-format guide. 283 tests pass.

Lesson: an agent's "done" is a claim, not evidence. Check it against the logs before you give the agent more autonomy.

What do you check before you trust an agent's "done"?

#AgenticAI #AIAgents #EvalDrivenDevelopment #VibeCoding #ClaudeCode #TheGenAcademy

---

_Before posting: tag The Gen Academy and the instructors, and attach 1–2 screenshots (Run detail of `sky-0904-0028` showing "💬 Agent CLAIMED" next to "🔍 Logs SHOW", and Step 2 with "📐 What should a policy look like?" open). Say that the logs are synthetic if you show them._
