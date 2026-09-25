// src/lib/backendClient.js
// -------------------------
// Real FastAPI/SSE integration for the CodeHeal backend.
//
// Public API
// ----------
//   runLivePipeline(scenario, handlers) -> { cancel() }
//
// handlers (identical contract to pipelineSimulator.js):
//   onPhase(phaseId)              'analyzing' | 'testing' | 'fixing' | 'verified' | 'failed'
//   onLog(line: string)           one terminal line at a time
//   onTestStatus(status)          'failing' | 'fixed'
//   onDiffReady({ before, after })
//   onDone()
//
// Scenario mapping (frontend id -> backend scenario name):
//   null-pointer    -> none_bug
//   broken-routing  -> broken_api
//   off-by-one      -> off_by_one

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

// In development the Vite proxy rewrites /api -> http://localhost:8000/api,
// so API_BASE is empty string (same-origin). For production builds, set the
// VITE_API_BASE_URL environment variable before `vite build`.
const API_BASE =
  typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_BASE_URL
    ? import.meta.env.VITE_API_BASE_URL
    : ''

// ---------------------------------------------------------------------------
// Scenario name mapping
// ---------------------------------------------------------------------------

const SCENARIO_MAP = {
  'null-pointer': 'none_bug',
  'broken-routing': 'broken_api',
  'off-by-one': 'off_by_one',
}

// ---------------------------------------------------------------------------
// Backend stage -> UI phase mapping
// ---------------------------------------------------------------------------

// Backend PipelineStage values (strings from PipelineStage.value):
//   idle | loading | running_tests | diagnosing | generating_test
//   applying_fix | verifying | complete | failed

function stageToPhase(stage) {
  switch (stage) {
    case 'loading':
    case 'diagnosing':
    case 'generating_test':
      return 'analyzing'
    case 'running_tests':
      return 'testing'
    case 'applying_fix':
    case 'verifying':
      return 'fixing'
    case 'complete':
      return 'verified'
    case 'failed':
      return 'failed'
    default:
      return null // 'idle' and unknown stages -- no UI phase change
  }
}

// ---------------------------------------------------------------------------
// Diff parser
// ---------------------------------------------------------------------------
// The backend sends a unified diff string (diff --git ...).
// Parse it into { before, after } that DiffViewer expects.

function parseDiff(diffStr) {
  if (!diffStr) return null

  const removedLines = []
  const addedLines = []

  for (const line of diffStr.split('\n')) {
    if (
      line.startsWith('---') ||
      line.startsWith('+++') ||
      line.startsWith('diff ') ||
      line.startsWith('index ') ||
      line.startsWith('@@')
    ) {
      continue
    }
    if (line.startsWith('-')) {
      removedLines.push(line.slice(1))
    } else if (line.startsWith('+')) {
      addedLines.push(line.slice(1))
    } else {
      // Context line -- appears in both sides.
      const ctx = line.startsWith(' ') ? line.slice(1) : line
      removedLines.push(ctx)
      addedLines.push(ctx)
    }
  }

  return {
    before: removedLines.join('\n'),
    after: addedLines.join('\n'),
  }
}

// ---------------------------------------------------------------------------
// Log line builder
// ---------------------------------------------------------------------------
// Turn the fields of a PipelineState payload into terminal lines, emitting
// only lines that haven't been seen before (tracked via the seenLogs Set).

function extractLogLines(data, seenLogs) {
  const lines = []

  function maybeAdd(line) {
    if (line && !seenLogs.has(line)) {
      seenLogs.add(line)
      lines.push(line)
    }
  }

  const stage = data.stage

  // Emit a human-readable phase banner the first time we enter each stage.
  const banners = {
    loading: '>> Loading repository...',
    running_tests: '>> Running initial test suite...',
    diagnosing: '>> Diagnosing root cause...',
    generating_test: '>> Generating reproducible test...',
    applying_fix: '>> Applying patch...',
    verifying: '>> Verifying fix...',
    complete: '>> Pipeline complete.',
    failed: '>> Pipeline failed.',
  }
  if (banners[stage]) maybeAdd(banners[stage])

  // Initial test output.
  const init = data.initial_test_result
  if (init) {
    const combined = ((init.stdout || '') + (init.stderr || '')).trim()
    if (combined) {
      for (const line of combined.split('\n')) maybeAdd(line)
    }
  }

  // Diagnosis.
  const diag = data.diagnosis
  if (diag) {
    if (diag.root_cause) maybeAdd('Root cause: ' + diag.root_cause)
    if (diag.affected_file) maybeAdd('Affected file: ' + diag.affected_file)
    if (diag.explanation) maybeAdd('Explanation: ' + diag.explanation)
    if (diag.confidence != null) maybeAdd('Confidence: ' + Math.round(diag.confidence * 100) + '%')
  }

  // Generated test.
  const genTest = data.generated_test
  if (genTest) {
    if (genTest.filename) maybeAdd('Generated test: ' + genTest.filename)
    if (genTest.confirmed_failing) maybeAdd('Generated test confirmed failing')
  }

  // Patch.
  const patch = data.patch
  if (patch) {
    if (patch.validated) maybeAdd('Patch validated')
  }

  // Final test output.
  const fin = data.final_test_result
  if (fin) {
    const combined = ((fin.stdout || '') + (fin.stderr || '')).trim()
    if (combined) {
      for (const line of combined.split('\n')) maybeAdd(line)
    }
  }

  // Error.
  if (data.error) maybeAdd('Error: ' + data.error)

  return lines
}

// ---------------------------------------------------------------------------
// Core SSE subscription
// ---------------------------------------------------------------------------

function subscribeToEvents(runId, handlers) {
  const { onPhase, onLog, onTestStatus, onDiffReady, onDone } = handlers
  const url = API_BASE + '/api/run/' + runId + '/events'
  const es = new EventSource(url)

  // Track which log lines have already been emitted so we never duplicate.
  const seenLogs = new Set()
  // Track last phase to avoid duplicate phase transitions.
  let lastPhase = null

  function handleState(data) {
    const stage = data.stage

    // Phase transition
    const phase = stageToPhase(stage)
    if (phase && phase !== lastPhase) {
      lastPhase = phase
      onPhase(phase)
    }

    // Test status
    if (stage === 'running_tests') {
      onTestStatus('failing')
    }
    if (stage === 'complete') {
      const finalResult = data.final_test_result
      const finalPassed = finalResult && finalResult.exit_code === 0
      onTestStatus(finalPassed ? 'fixed' : 'failing')
    }

    // Diff
    if (stage === 'complete' || stage === 'applying_fix' || stage === 'verifying') {
      const diffStr = data.patch && data.patch.diff
      if (diffStr) {
        const parsed = parseDiff(diffStr)
        if (parsed) onDiffReady(parsed)
      }
    }

    // Log lines
    for (const line of extractLogLines(data, seenLogs)) {
      onLog(line)
    }
  }

  es.addEventListener('pipeline_state', function(e) {
    try {
      const payload = JSON.parse(e.data)
      handleState(payload.data != null ? payload.data : payload)
    } catch (_) {
      // Malformed event -- skip silently.
    }
  })

  es.addEventListener('complete', function(e) {
    try {
      const payload = JSON.parse(e.data)
      handleState(payload.data != null ? payload.data : payload)
    } catch (_) { /* ignore */ }
    es.close()
    onDone()
  })

  es.addEventListener('failed', function(e) {
    try {
      const payload = JSON.parse(e.data)
      const data = payload.data != null ? payload.data : payload
      onPhase('failed')
      const errMsg = data.error || 'Backend pipeline failed.'
      if (!seenLogs.has(errMsg)) {
        seenLogs.add(errMsg)
        onLog('Error: ' + errMsg)
      }
    } catch (_) {
      onLog('Error: Pipeline failed (unknown error).')
    }
    es.close()
    onDone()
  })

  es.addEventListener('stream_end', function() {
    es.close()
  })

  es.onerror = function() {
    onLog('Error: Lost connection to backend.')
    es.close()
    onDone()
  }

  return {
    cancel: function() {
      es.close()
    },
  }
}

// ---------------------------------------------------------------------------
// Public entry point -- same signature as runSimulatedPipeline
// ---------------------------------------------------------------------------

export async function runLivePipeline(scenario, handlers) {
  const { onLog, onDone } = handlers
  const backendScenario = SCENARIO_MAP[scenario.id]

  if (!backendScenario) {
    onLog('Error: Unknown scenario "' + scenario.id + '".')
    onDone()
    return { cancel: function() {} }
  }

  var subscription = null

  try {
    const resp = await fetch(API_BASE + '/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scenario: backendScenario }),
    })

    if (!resp.ok) {
      var detail = resp.statusText
      try {
        const body = await resp.json()
        if (body.detail) detail = body.detail
      } catch (_) { /* ignore */ }
      onLog('Error: POST /api/run failed (' + resp.status + '): ' + detail)
      onDone()
      return { cancel: function() {} }
    }

    const json = await resp.json()
    subscription = subscribeToEvents(json.run_id, handlers)
  } catch (err) {
    // Network error -- backend is unreachable.
    onLog('Error: Cannot reach the CodeHeal backend. Is it running on ' + (API_BASE || 'http://localhost:8000') + '?')
    onLog('Detail: ' + err.message)
    onDone()
    return { cancel: function() {} }
  }

  return subscription
}