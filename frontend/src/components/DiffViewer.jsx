import './DiffViewer.css'

export default function DiffViewer({ file, diff }) {
  const beforeLines = diff ? diff.before.split('\n') : []
  const afterLines = diff ? diff.after.split('\n') : []

  return (
    <section className="diff-panel">
      <div className="diff-panel-header">
        <h2>Code Diff</h2>
        {diff && <span className="diff-file">{file}</span>}
      </div>

      {!diff ? (
        <div className="diff-empty">
          <p>No fix applied yet.</p>
          <small>The diff appears once the Refactoring Subagent patches the file.</small>
        </div>
      ) : (
        <div className="diff-columns">
          <div className="diff-column">
            <div className="diff-column-label diff-column-label--before">Before</div>
            <pre className="diff-code">
              {beforeLines.map((line, i) => (
                <div className="diff-line diff-line--removed" key={i}>
                  <span className="diff-gutter">−</span>
                  <span>{line}</span>
                </div>
              ))}
            </pre>
          </div>
          <div className="diff-column">
            <div className="diff-column-label diff-column-label--after">After</div>
            <pre className="diff-code">
              {afterLines.map((line, i) => (
                <div className="diff-line diff-line--added" key={i}>
                  <span className="diff-gutter">+</span>
                  <span>{line}</span>
                </div>
              ))}
            </pre>
          </div>
        </div>
      )}
    </section>
  )
}
