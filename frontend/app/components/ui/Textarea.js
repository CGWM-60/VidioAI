export default function Textarea({
  label,
  help,
  error,
  maxLength,
  value = "",
  id,
  rows = 5,
  className = "",
  ...props
}) {
  const length = typeof value === "string" ? value.length : 0;

  return (
    <div className={`vidio-field ${className}`}>
      {label && (
        <label htmlFor={id} className="vidio-label">
          {label}
        </label>
      )}

      <div className={`vidio-textarea-wrapper ${error ? "vidio-input-wrapper-error" : ""}`}>
        <textarea
          id={id}
          rows={rows}
          maxLength={maxLength}
          value={value}
          className="vidio-textarea"
          {...props}
        />

        {maxLength && (
          <div className="vidio-textarea-counter">
            {length} / {maxLength}
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
