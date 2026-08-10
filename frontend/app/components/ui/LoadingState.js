export default function LoadingState({
  title = "Chargement...",
  description,
  compact = false,
}) {
  return (
    <div className={`vidio-state ${compact ? "vidio-state-compact" : ""}`}>
      <div className="vidio-loader">
        <span />
        <span />
        <span />
      </div>

      <h3>{title}</h3>

      {description && <p>{description}</p>}
    </div>
  );
}
