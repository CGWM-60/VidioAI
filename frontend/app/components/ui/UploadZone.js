"use client";

import { useRef, useState } from "react";

export default function UploadZone({
  accept,
  multiple = false,
  disabled = false,
  onFiles,
  title = "Glissez-déposez un fichier ici",
  description = "ou cliquez pour parcourir",
  helper,
  icon = "⇧",
  className = "",
}) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  function emitFiles(fileList) {
    if (!fileList || disabled) return;

    let files = Array.from(fileList);

    if (!multiple) {
      files = files.slice(0, 1);
    }

    onFiles?.(files);
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragging(false);
    emitFiles(event.dataTransfer.files);
  }

  return (
    <div
      className={[
        "upload-zone",
        dragging ? "upload-zone-active" : "",
        disabled ? "upload-zone-disabled" : "",
        className,
      ].join(" ").trim()}
      role="button"
      aria-label={`${title}. ${description}`}
      tabIndex={disabled ? -1 : 0}
      onClick={() => !disabled && inputRef.current?.click()}
      onKeyDown={(event) => {
        if (!disabled && (event.key === "Enter" || event.key === " ")) {
          event.preventDefault();
          inputRef.current?.click();
        }
      }}
      onDragOver={(event) => {
        event.preventDefault();
        if (!disabled) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
    >
      <input
        ref={inputRef}
        type="file"
        hidden
        accept={accept}
        multiple={multiple}
        disabled={disabled}
        onChange={(event) => {
          emitFiles(event.target.files);
          event.target.value = "";
        }}
      />

      <div className="upload-zone-icon">
        {icon}
      </div>

      <strong className="upload-zone-title">
        {title}
      </strong>

      <span className="upload-zone-description">
        {description}
      </span>

      {helper && (
        <span className="upload-zone-helper">
          {helper}
        </span>
      )}
    </div>
  );
}
