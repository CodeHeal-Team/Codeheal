import { useEffect, useRef, useState } from 'react'
import { scenarios } from '../data/scenarios.js'
import './ScenarioSelector.css'

export default function ScenarioSelector({ selectedId, onSelect }) {
  const [open, setOpen] = useState(false)
  const rootRef = useRef(null)

  const selected = scenarios.find((s) => s.id === selectedId) ?? scenarios[0]

  // Close the dropdown when clicking anywhere outside it.
  useEffect(() => {
    function handleClickOutside(event) {
      if (rootRef.current && !rootRef.current.contains(event.target)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  function handlePick(scenario) {
    onSelect(scenario.id)
    setOpen(false)
  }

  return (
    <div className="scenario-selector" ref={rootRef}>
      <button
        className="scenario-trigger"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <div className="scenario-trigger-text">
          <span className="scenario-trigger-label">{selected.label}</span>
          <span className="scenario-trigger-file">{selected.file}</span>
        </div>
        <svg
          className={`scenario-chevron ${open ? 'is-open' : ''}`}
          width="14"
          height="14"
          viewBox="0 0 14 14"
          fill="none"
        >
          <path d="M3 5.5L7 9.5L11 5.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>

      {open && (
        <div className="scenario-panel">
          {scenarios.map((scenario) => (
            <button
              key={scenario.id}
              className={`scenario-option ${scenario.id === selected.id ? 'is-selected' : ''}`}
              onClick={() => handlePick(scenario)}
            >
              <div className="scenario-option-top">
                <span className="scenario-option-label">{scenario.label}</span>
                <span className="scenario-option-tag">{scenario.tag}</span>
              </div>
              <p className="scenario-option-desc">{scenario.description}</p>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
