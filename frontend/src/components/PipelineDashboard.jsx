import './PipelineDashboard.css'

const STEPS = [
  { id: 'idle', label: 'Idle' },
  { id: 'analyzing', label: 'Analyzing' },
  { id: 'testing', label: 'Testing' },
  { id: 'fixing', label: 'Fixing' },
  { id: 'verified', label: 'Verified' },
]

function stepStatus(stepId, currentPhase) {
  const order = STEPS.map((s) => s.id)
  const currentIndex = order.indexOf(currentPhase)
  const stepIndex = order.indexOf(stepId)
  if (stepIndex < currentIndex) return 'done'
  if (stepIndex === currentIndex) return currentPhase === 'idle' ? 'pending' : 'active'
  return 'pending'
}

function StatusBadge({ testStatus }) {
  if (testStatus === 'failing') {
    return (
      <div className="status-badge status-badge--fail">
        <span className="status-badge-icon">❌</span>
        <span>Before: Test Failed</span>
      </div>
    )
  }
  if (testStatus === 'fixed') {
    return (
      <div className="status-badge status-badge--pass">
        <span className="status-badge-icon">✅</span>
        <span>After: Test Passed</span>
      </div>
    )
  }
  return (
    <div className="status-badge status-badge--idle">
      <span className="status-badge-icon">○</span>
      <span>Awaiting run</span>
    </div>
  )
}

export default function PipelineDashboard({ phase, testStatus, running, onRun }) {
  const fixing = phase === 'fixing'

  return (
    <section className="pipeline-panel">
      <div className="pipeline-panel-header">
        <h2>Pipeline</h2>
        <button className="run-button" onClick={onRun} disabled={running}>
          {running ? (
            <>
              <span className="run-spinner" />
              Running…
            </>
          ) : (
            'Run CodeHeal'
          )}
        </button>
      </div>

      <div className="stepper">
        {STEPS.map((step, i) => {
          const status = stepStatus(step.id, phase)
          return (
            <div className="stepper-item" key={step.id}>
              <div className="stepper-node-row">
                <div className={`stepper-node stepper-node--${status}`}>
                  {status === 'done' ? '✓' : i + 1}
                </div>
                {i < STEPS.length - 1 && (
                  <div className={`stepper-line ${status === 'done' ? 'is-done' : ''}`} />
                )}
              </div>
              <span className={`stepper-label stepper-label--${status}`}>{step.label}</span>
            </div>
          )
        })}
      </div>

      <div className="pipeline-panel-footer">
        <StatusBadge testStatus={testStatus} />
        {fixing && (
          <div className="status-badge status-badge--progress">
            <span className="status-badge-icon">🔧</span>
            <span>In Progress: Fix Applied</span>
          </div>
        )}
      </div>
    </section>
  )
}
