"use client";

import { useEffect, useRef } from "react";

export default function Modal({
  open,
  onClose,
  title,
  subtitle,
  children,
  footer,
  size = "md",
  closeOnBackdrop = true,
}) {
  const closeButtonRef = useRef(null);

  useEffect(() => {
    if (!open) return;

    const previouslyFocused = document.activeElement;

    function handleEscape(event) {
      if (event.key === "Escape") {
        onClose?.();
      }
    }

    document.body.style.overflow = "hidden";
    document.addEventListener("keydown", handleEscape);
    closeButtonRef.current?.focus();

    return () => {
      document.body.style.overflow = "";
      document.removeEventListener("keydown", handleEscape);
      previouslyFocused?.focus?.();
    };
  }, [open, onClose]);

  if (!open) return null;

  function handleBackdrop(event) {
    if (closeOnBackdrop && event.target === event.currentTarget) {
      onClose?.();
    }
  }

  return (
    <div className="vidio-modal-backdrop" onMouseDown={handleBackdrop}>
      <div
        className={`vidio-modal vidio-modal-${size}`}
        role="dialog"
        aria-modal="true"
        aria-label={title || "Fenêtre"}
      >
        <div className="vidio-modal-header">
          <div>
            {title && <h2>{title}</h2>}
            {subtitle && <p>{subtitle}</p>}
          </div>

          <button
            ref={closeButtonRef}
            type="button"
            className="vidio-modal-close"
            onClick={onClose}
            aria-label="Fermer"
          >
            ×
          </button>
        </div>

        <div className="vidio-modal-body">
          {children}
        </div>

        {footer && (
          <div className="vidio-modal-footer">
            {footer}
          </div>
        )}
      </div>
    </div>
  );
}
