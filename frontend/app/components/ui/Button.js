"use client";

export default function Button({
  children,
  variant = "primary",
  size = "md",
  icon,
  loading = false,
  disabled = false,
  fullWidth = false,
  className = "",
  type = "button",
  ...props
}) {
  const variants = {
    primary: "vidio-button-primary",
    secondary: "vidio-button-secondary",
    danger: "vidio-button-danger",
    ghost: "vidio-button-ghost",
  };

  return (
    <button
      type={type}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={[
        "vidio-button",
        variants[variant] || variants.primary,
        `vidio-button-${size}`,
        fullWidth ? "w-100" : "",
        className,
      ].join(" ").trim()}
      {...props}
    >
      {loading ? (
        <span className="vidio-button-spinner" aria-hidden="true" />
      ) : (
        icon
      )}

      {children && <span>{children}</span>}
    </button>
  );
}
