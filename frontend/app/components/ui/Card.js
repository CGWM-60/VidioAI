export default function Card({
  children,
  className = "",
  interactive = false,
  selected = false,
  padding = true,
  ...props
}) {
  return (
    <div
      className={[
        "vidio-card",
        interactive ? "vidio-card-interactive" : "",
        selected ? "vidio-card-selected" : "",
        padding ? "p-3" : "",
        className,
      ].join(" ").trim()}
      {...props}
    >
      {children}
    </div>
  );
}
