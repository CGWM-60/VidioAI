export default function CircularProgress({
  value = 0,
  max = 100,
  size = 220,
  strokeWidth = 14,
  label,
  sublabel,
  className = "",
}) {
  const safeMax = max > 0 ? max : 100;
  const percent = Math.max(0, Math.min(100, (value / safeMax) * 100));

  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const dashOffset = circumference - (percent / 100) * circumference;

  return (
    <div className={`vidio-circular-wrap ${className}`}>
      <div
        className="vidio-circular-progress"
        style={{ width: size, height: size }}
        role="progressbar"
        aria-label={label || "Progression"}
        aria-valuenow={Math.round(percent)}
        aria-valuemin="0"
        aria-valuemax="100"
      >
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
          <circle
            className="vidio-circular-track"
            cx={size / 2}
            cy={size / 2}
            r={radius}
            strokeWidth={strokeWidth}
          />

          <circle
            className="vidio-circular-value"
            cx={size / 2}
            cy={size / 2}
            r={radius}
            strokeWidth={strokeWidth}
            strokeDasharray={circumference}
            strokeDashoffset={dashOffset}
          />
        </svg>

        <div className="vidio-circular-center">
          <strong>{Math.round(percent)}%</strong>
        </div>
      </div>

      {label && <div className="vidio-circular-label">{label}</div>}
      {sublabel && <div className="vidio-circular-sublabel">{sublabel}</div>}
    </div>
  );
}
