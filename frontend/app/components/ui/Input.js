export default function Input({
  label,
  help,
  error,
  icon,
  rightElement,
  id,
  className = "",
  inputClassName = "",
  ...props
}) {
  return (
    <div className={`vidio-field ${className}`}>
      {label && (
        <label htmlFor={id} className="vidio-label">
          {label}
        </label>
      )}

      <div className={`vidio-input-wrapper ${error ? "vidio-input-wrapper-error" : ""}`}>
        {icon && <span className="vidio-input-icon">{icon}</span>}

        <input
          id={id}
          className={`vidio-input ${icon ? "vidio-input-with-icon" : ""} ${inputClassName}`}
          {...props}
        />

        {rightElement && (
          <div className="vidio-input-right">
            {rightElement}
          </div>
        )}
      </div>

      {error ? (
        <div className="vidio-field-error">{error}</div>
      ) : help ? (
        <div className="vidio-field-help">{help}</div>
      ) : null}
    </div>
  );
}
