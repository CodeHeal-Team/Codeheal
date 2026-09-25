# CodeHeal — Command Center (Part 3)

Interactive dashboard for CodeHeal: pick a bug scenario, click **Run CodeHeal**,
and watch the self-healing pipeline (Analyzing → Testing → Fixing → Verified)
stream live logs and produce a before/after code diff.

## Run it

```
npm install
npm run dev
```

Open the printed local URL (usually http://localhost:5173).

## What's here

- `src/data/scenarios.js` — the 3 pre-configured bugs (Null Pointer, Broken
  API Routing, Off-by-One), each with before/after code and per-phase log lines.
- `src/components/ScenarioSelector.jsx` — the dropdown bug picker in the header.
- `src/components/PipelineDashboard.jsx` — the 5-step tracker (Idle → Analyzing
  → Testing → Fixing → Verified) plus the ❌ / 🔧 / ✅ status badges, and the
  "Run CodeHeal" button.
- `src/components/TerminalLog.jsx` — the live-streaming terminal panel.
- `src/components/DiffViewer.jsx` — the side-by-side before/after code diff.
- `src/lib/pipelineSimulator.js` — drives the whole demo locally (no backend
  needed) by emitting the same events a real backend would.
- `src/lib/backendClient.js` — **read this when Part 1's backend is ready.**
  It documents exactly where to swap the simulator for a real WebSocket or
  REST connection to Part 1 (Core Brain), with the handler contract already
  matching so nothing else in the UI has to change.

## Wiring in the real backend (Parts 1 & 2)

Everything currently runs off `runSimulatedPipeline`. When Part 1 exposes a
real endpoint:

1. Implement `runLivePipeline(scenario, handlers)` in `src/lib/backendClient.js`
   (a WebSocket sketch is already commented in there).
2. In `App.jsx`, change the import from `pipelineSimulator.js` to
   `backendClient.js`.

Nothing else changes — every component already reacts to `phase`,
`testStatus`, `logs`, and `diff` state, regardless of where those values
come from.
