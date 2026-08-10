export default function Panel({
  children,
  title,
  subtitle,
  icon,
  actions,
  className = "",
  bodyClassName = "",
}) {
  return (
    <section className={`vidio-panel ${className}`}>
      {(title || subtitle || icon || actions) && (
        <div className="vidio-panel-header">
          <div className="vidio-panel-heading">
            {icon && (
              <div className="vidio-panel-icon">
                {icon}
              </div>
            )}

            <div>
              {title && <h2 className="vidio-panel-title">{title}</h2>}
              {subtitle && <p className="vidio-panel-subtitle">{subtitle}</p>}
            </div>
          </div>

          {actions && (
            <div className="vidio-panel-actions">
              {actions}
            </div>
          )}
        </div>
      )}

      <div className={`vidio-panel-body ${bodyClassName}`}>
        {children}
      </div>
    </section>
  );
}
