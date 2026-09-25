import { useEffect, useRef, useState } from 'react'
import ScenarioSelector from './components/ScenarioSelector.jsx'
import PipelineDashboard from './components/PipelineDashboard.jsx'
import TerminalLog from './components/TerminalLog.jsx'
import DiffViewer from './components/DiffViewer.jsx'
import { scenarios } from './data/scenarios.js'
import { runLivePipeline } from './lib/backendClient.js'
import './App.css'

export default function App() {
  const [scenarioId, setScenarioId] = useState(scenarios[0].id)
  const [phase, setPhase] = useState('idle')
  const [testStatus, setTestStatus] = useState('idle')
  const [logs, setLogs] = useState([])
  const [diff, setDiff] = useState(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState(null)

  const pipelineRef = useRef(null)
  const scenario = scenarios.find((s) => s.id === scenarioId)

  // Reset the dashboard whenever a different bug scenario is picked.
  function resetRun() {
    pipelineRef.current?.cancel()
    setPhase('idle')
    setTestStatus('idle')
    setLogs([])
    setDiff(null)
    setRunning(false)
    setError(null)
  }

  function handleSelectScenario(id) {
    setScenarioId(id)
    resetRun()
  }

  async function handleRun() {
    resetRun()
    setRunning(true)

    const handle = await runLivePipeline(scenario, {
      onPhase: setPhase,
      onLog: (line) => setLogs((prev) => [...prev, line]),
      onTestStatus: setTestStatus,
      onDiffReady: setDiff,
      onDone: () => setRunning(false),
    })

    // runLivePipeline returns after POST completes (or fails immediately);
    // the handle lets us cancel the SSE stream if the user switches scenario
    // or the component unmounts before the pipeline finishes.
    pipelineRef.current = handle
  }

  // Cancel any in-flight run if the component unmounts.
  useEffect(() => () => pipelineRef.current?.cancel(), [])

  return (
    <div className="shell">
      <header className="shell-header">
        <div className="brand">
          <span className="brand-mark" />
          <h1>CodeHeal</h1>
          <span className="brand-sub">Command Center</span>
        </div>
        <ScenarioSelector selectedId={scenarioId} onSelect={handleSelectScenario} />
      </header>

      {error && (
        <div className="error-banner" role="alert">
          {error}
        </div>
      )}

      <main className="shell-body">
        <PipelineDashboard
          phase={phase}
          testStatus={testStatus}
          running={running}
          onRun={handleRun}
        />

        <div className="shell-split">
          <TerminalLog lines={logs} running={running} />
          <DiffViewer file={scenario.file} diff={diff} />
        </div>
      </main>
    </div>
  )
}
