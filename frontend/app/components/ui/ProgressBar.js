export default function ProgressBar({
  value = 0,
  max = 100,
  label,
  showValue = false,
  size = "md",
  className = "",
}) {
  const safeMax = max > 0 ? max : 100;
  const percent = Math.max(0, Math.min(100, (value / safeMax) * 100));

  return (
    <div className={className}>
      {(label || showValue) && (
        <div className="vidio-progress-header">
          <span>{label}</span>
          {showValue && <span>{Math.round(percent)}%</span>}
        </div>
      )}

      <div
        className={`progress-track progress-track-${size}`}
        role="progressbar"
        aria-label={label || "Progression"}
        aria-valuenow={Math.round(percent)}
        aria-valuemin="0"
        aria-valuemax={safeMax}
      >
        <div
          className="progress-value"
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}
