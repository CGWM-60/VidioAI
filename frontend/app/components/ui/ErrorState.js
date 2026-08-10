export default function ErrorState({
  title = "Une erreur est survenue",
  description = "Impossible de terminer l'opération.",
  action,
  compact = false,
}) {
  return (
    <div className={`vidio-state vidio-state-error ${compact ? "vidio-state-compact" : ""}`}>
      <div className="vidio-state-icon">
        !
      </div>

      <h3>{title}</h3>

      <p>{description}</p>

      {action && (
        <div className="mt-3">
          {action}
        </div>
      )}
    </div>
  );
}
