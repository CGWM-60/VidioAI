"use client";

export default function Tabs({
  tabs = [],
  active,
  onChange,
  fluid = false,
  className = "",
}) {
  return (
    <div
      className={`vidio-tabs ${fluid ? "vidio-tabs-fluid" : ""} ${className}`}
      role="tablist"
    >
      {tabs.map((tab) => (
        <button
          key={tab.id}
          type="button"
          role="tab"
          aria-selected={active === tab.id}
          onClick={() => onChange?.(tab.id)}
          className={`vidio-tab ${active === tab.id ? "vidio-tab-active" : ""}`}
        >
          {tab.icon && <span className="vidio-tab-icon">{tab.icon}</span>}

          <span className="vidio-tab-text">
            <span>{tab.label}</span>
            {tab.subtitle && (
              <small>{tab.subtitle}</small>
            )}
          </span>

          {active === tab.id && tab.showCheck && (
            <span className="vidio-tab-check">✓</span>
          )}
        </button>
      ))}
    </div>
  );
}
