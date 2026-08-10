export default function EmptyState({
  icon = "＋",
  title = "Aucun élément",
  description = "Il n'y a encore rien à afficher.",
  action,
  compact = false,
}) {
  return (
    <div className={`vidio-state ${compact ? "vidio-state-compact" : ""}`}>
      <div className="vidio-state-icon">
        {icon}
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
