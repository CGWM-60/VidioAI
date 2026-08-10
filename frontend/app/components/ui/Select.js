export default function Select({
  label,
  help,
  error,
  options = [],
  placeholder,
  icon,
  id,
  className = "",
  ...props
}) {
  return (
    <div className={`vidio-field ${className}`}>
      {label && (
        <label htmlFor={id} className="vidio-label">
          {label}
        </label>
      )}

      <div className={`vidio-select-wrapper ${error ? "vidio-input-wrapper-error" : ""}`}>
        {icon && <span className="vidio-input-icon">{icon}</span>}

        <select
          id={id}
          className={`vidio-select ${icon ? "vidio-select-with-icon" : ""}`}
          {...props}
        >
          {placeholder && <option value="">{placeholder}</option>}

          {options.map((option) => {
            const value = typeof option === "object" ? option.value : option;
            const text = typeof option === "object" ? option.label : option;
            const disabled = typeof option === "object" ? option.disabled : false;

            return (
              <option key={value} value={value} disabled={disabled}>
                {text}
              </option>
            );
          })}
        </select>
      </div>

      {error ? (
        <div className="vidio-field-error">{error}</div>
      ) : help ? (
        <div className="vidio-field-help">{help}</div>
      ) : null}
    </div>
  );
}
