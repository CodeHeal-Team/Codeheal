import { useEffect, useRef } from 'react'
import './TerminalLog.css'

function lineKind(line) {
  if (/^FAIL/.test(line) || /Error/.test(line) || line.startsWith('  ●')) return 'fail'
  if (/^PASS/.test(line) || line.trim().startsWith('✓')) return 'pass'
  if (line.startsWith('$')) return 'command'
  return 'info'
}

export default function TerminalLog({ lines, running }) {
  const scrollRef = useRef(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [lines])

  return (
    <section className="terminal-panel">
      <div className="terminal-panel-header">
        <h2>Terminal</h2>
        <div className="terminal-dots">
          <span />
          <span />
          <span />
        </div>
      </div>

      <div className="terminal-body" ref={scrollRef}>
        {lines.length === 0 && (
          <p className="terminal-empty">Run CodeHeal to stream live output here.</p>
        )}
        {lines.map((line, i) => (
          <div key={i} className={`terminal-line terminal-line--${lineKind(line)}`}>
            {line}
          </div>
        ))}
        {running && <span className="terminal-cursor" />}
      </div>
    </section>
  )
}
