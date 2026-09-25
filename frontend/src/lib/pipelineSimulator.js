// Simulates the pipeline that Part 1 (Core Brain) + Part 2 (Agent
// Intelligence) will actually run. It drives the same events a real
// backend would emit, at demo-friendly speed, so Part 3 can be built
// and demoed independently of Parts 1 and 2.
//
// --- Integration point for whoever owns Part 1 ---
// When the real backend is ready, replace the call to `runSimulatedPipeline`
// in App.jsx with `runLivePipeline` from backendClient.js. Both functions
// share the exact same handlers signature below, so nothing else in the
// UI needs to change.
//
// handlers:
//   onPhase(phaseId)              -- 'analyzing' | 'testing' | 'fixing' | 'verified'
//   onLog(line: string)           -- one terminal line at a time
//   onTestStatus(status)          -- 'failing' | 'fixed'
//   onDiffReady({ before, after })
//   onDone()

const STEP_DELAY = 550

export function runSimulatedPipeline(scenario, handlers) {
  let cancelled = false
  const { onPhase, onLog, onTestStatus, onDiffReady, onDone } = handlers

  async function run() {
    const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms))
    const emitLines = async (lines) => {
      for (const line of lines) {
        if (cancelled) return
        onLog(line)
        await wait(STEP_DELAY)
      }
    }

    if (cancelled) return
    onPhase('analyzing')
    await emitLines(scenario.logs.analyzing)

    if (cancelled) return
    onPhase('testing')
    onTestStatus('failing')
    await emitLines(scenario.logs.testingInitial)

    if (cancelled) return
    onPhase('fixing')
    await emitLines(scenario.logs.fixing)
    onDiffReady(scenario.code)

    if (cancelled) return
    await emitLines(scenario.logs.verifying)
    onTestStatus('fixed')

    if (cancelled) return
    onPhase('verified')
    onDone()
  }

  run()

  return {
    cancel() {
      cancelled = true
    },
  }
}
