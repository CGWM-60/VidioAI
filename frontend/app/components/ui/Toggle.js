"use client";

export default function Toggle({
  checked = false,
  onChange,
  label,
  description,
  disabled = false,
  className = "",
}) {
  function handleClick() {
    if (!disabled) {
      onChange?.(!checked);
    }
  }

  return (
    <div className={`vidio-toggle-row ${className}`}>
      {(label || description) && (
        <div className="vidio-toggle-copy">
          {label && <div className="vidio-toggle-label">{label}</div>}
          {description && <div className="vidio-toggle-description">{description}</div>}
        </div>
      )}

      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={label || "Option"}
        disabled={disabled}
        onClick={handleClick}
        className={`switch ${checked ? "switch-active" : ""}`}
      >
        <span className="visually-hidden">
          {checked ? "Activé" : "Désactivé"}
        </span>
      </button>
    </div>
  );
}
