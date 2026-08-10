"use client";

import { useEffect } from "react";

export default function Toast({
  open,
  onClose,
  title,
  children,
  variant = "info",
  duration = 4000,
}) {
  useEffect(() => {
    if (!open || !duration) return;

    const timer = setTimeout(() => {
      onClose?.();
    }, duration);

    return () => clearTimeout(timer);
  }, [open, duration, onClose]);

  if (!open) return null;

  return (
    <div
      className={`vidio-toast vidio-toast-${variant}`}
      role={variant === "error" ? "alert" : "status"}
      aria-live={variant === "error" ? "assertive" : "polite"}
    >
      <div className="vidio-toast-marker" />

      <div className="flex-grow-1">
        {title && <div className="vidio-toast-title">{title}</div>}
        <div className="vidio-toast-message">{children}</div>
      </div>

      <button
        type="button"
        onClick={onClose}
        className="vidio-toast-close"
        aria-label="Fermer"
      >
        ×
      </button>
    </div>
  );
}
