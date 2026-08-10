export default function Badge({
  children,
  variant = "purple",
  dot = false,
  className = "",
}) {
  const variants = {
    purple: "badge-purple",
    green: "badge-green",
    red: "badge-red",
    warning: "badge-warning",
    info: "badge-info",
    gray: "badge-gray",
  };

  return (
    <span className={`badge ${variants[variant] || variants.purple} ${className}`}>
      {dot && <span className="vidio-badge-dot" />}
      {children}
    </span>
  );
}
