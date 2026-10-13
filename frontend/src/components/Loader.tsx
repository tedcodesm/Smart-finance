/** Circuit-trace SVG loader (from Uiverse.io by Vosoone). */

export default function Loader({ label }: { label?: string }) {
  return (
    <div className="loader-wrapper" role="status" aria-label={label ?? "Loading"}>
      <div className="main-container">
        <svg
          className="loader"
          viewBox="0 0 200 100"
          xmlns="http://www.w3.org/2000/svg"
        >
          {/* ── Chip body ── */}
          <rect
            x="70" y="25" width="60" height="50"
            rx="8" ry="8"
            fill="#1a1a2e"
            stroke="#333"
            strokeWidth="1"
          />

          {/* ── Chip pins (left side) ── */}
          {[35, 43, 51, 59, 67].map((y, i) => (
            <rect key={`l${i}`} className="chip-pin" x="58" y={y} width="12" height="4" rx="1" fill="#2a2a3e" />
          ))}

          {/* ── Chip pins (right side) ── */}
          {[35, 43, 51, 59, 67].map((y, i) => (
            <rect key={`r${i}`} className="chip-pin" x="130" y={y} width="12" height="4" rx="1" fill="#2a2a3e" />
          ))}

          {/* ── Chip label ── */}
          <text
            x="100" y="48"
            textAnchor="middle"
            className="chip-text"
            fill="#6d4aff"
            fontSize="6"
            fontFamily="monospace"
          >
            MERCY
          </text>
          <text
            x="100" y="57"
            textAnchor="middle"
            fill="#555"
            fontSize="4"
            fontFamily="monospace"
          >
            AI·FIN
          </text>

          {/* ── Trace backgrounds ── */}
          <path className="trace-bg" d="M 10 37 H 58" />
          <path className="trace-bg" d="M 10 51 H 58" />
          <path className="trace-bg" d="M 10 63 H 58" />
          <path className="trace-bg" d="M 142 37 H 190" />
          <path className="trace-bg" d="M 142 51 H 190" />
          <path className="trace-bg" d="M 142 63 H 190" />

          {/* ── Animated trace flows ── */}
          <path className="trace-flow yellow" d="M 10 37 H 58" />
          <path className="trace-flow blue"   d="M 10 51 H 58" style={{ animationDelay: "0.4s" }} />
          <path className="trace-flow green"  d="M 10 63 H 58" style={{ animationDelay: "0.8s" }} />
          <path className="trace-flow purple" d="M 142 37 H 190" style={{ animationDelay: "0.2s" }} />
          <path className="trace-flow red"    d="M 142 51 H 190" style={{ animationDelay: "0.6s" }} />
          <path className="trace-flow blue"   d="M 142 63 H 190" style={{ animationDelay: "1.0s" }} />
        </svg>
      </div>
      {label && <p className="loader-label">{label}</p>}
    </div>
  );
}
