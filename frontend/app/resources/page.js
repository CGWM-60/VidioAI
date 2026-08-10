"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  BsActivity,
  BsArrowClockwise,
  BsCheckCircleFill,
  BsClockHistory,
  BsCpu,
  BsDeviceSsd,
  BsExclamationTriangleFill,
  BsGpuCard,
  BsHddStack,
  BsMemory,
  BsPcDisplayHorizontal,
  BsServer,
  BsThermometerHalf,
} from "react-icons/bs";
import {
  Button,
  CircularProgress,
  ErrorState,
  LoadingState,
  ProgressBar,
} from "../components/ui";
import styles from "./resources.module.css";
import { API_BASE_URL } from "../lib/api";

const RESOURCES_ENDPOINT = `${API_BASE_URL}/api/resources`;
const REFRESH_INTERVAL = 15_000;

function formatBytes(value) {
  if (value === null || value === undefined) return "Non disponible";
  const bytes = Number(value);

  if (!Number.isFinite(bytes) || bytes < 0) return "—";
  if (bytes === 0) return "0 octet";

  const units = ["octets", "Ko", "Mo", "Go", "To", "Po"];
  const unitIndex = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1,
  );
  const converted = bytes / 1024 ** unitIndex;
  const decimals = converted >= 100 || unitIndex === 0 ? 0 : 1;

  return `${converted.toLocaleString("fr-FR", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })} ${units[unitIndex]}`;
}

function getPercent(used, total) {
  const safeUsed = Number(used);
  const safeTotal = Number(total);

  if (!Number.isFinite(safeUsed) || !Number.isFinite(safeTotal) || safeTotal <= 0) {
    return 0;
  }

  return Math.max(0, Math.min(100, (safeUsed / safeTotal) * 100));
}

function formatTemperature(value) {
  if (value === null || value === undefined) return "Non disponible";
  const temperature = Number(value);
  return Number.isFinite(temperature) && temperature > 0
    ? `${Math.round(temperature)} °C`
    : "Non disponible";
}

function formatFrequency(value) {
  const frequency = Number(value);

  if (!Number.isFinite(frequency) || frequency <= 0) return "Non disponible";
  if (frequency < 1000) return `${Math.round(frequency)} MHz`;

  return `${(frequency / 1000).toLocaleString("fr-FR", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })} GHz`;
}

function formatUptime(value) {
  if (value === null || value === undefined) return "Non disponible";
  const seconds = Math.max(0, Number(value) || 0);
  const days = Math.floor(seconds / 86_400);
  const hours = Math.floor((seconds % 86_400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainingSeconds = Math.floor(seconds % 60);

  return [
    days > 0 ? `${days}j` : null,
    `${hours}h`,
    `${minutes}m`,
    `${remainingSeconds}s`,
  ].filter(Boolean).join(" ");
}

function formatDateTime(value) {
  const timestamp = Number(value);
  if (!Number.isFinite(timestamp) || timestamp <= 0) return "Non disponible";

  return new Date(timestamp * 1000).toLocaleString("fr-FR", {
    dateStyle: "medium",
    timeStyle: "medium",
  });
}

function InfoRow({ label, value, children }) {
  return (
    <div className={styles.infoRow}>
      <span>{label}</span>
      <div className={styles.infoValue}>
        <strong>{value}</strong>
        {children}
      </div>
    </div>
  );
}

function Metric({ label, value, icon }) {
  return (
    <div className={styles.metric}>
      <span className={styles.metricIcon} aria-hidden="true">
        {icon}
      </span>
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </div>
  );
}

function ResourcePanel({ title, icon, children, className = "" }) {
  return (
    <section className={`${styles.panel} ${className}`}>
      <h2 className={styles.panelTitle}>
        <span aria-hidden="true">{icon}</span>
        {title}
      </h2>
      {children}
    </section>
  );
}

export default function ResourcesPage() {
  const [profile, setProfile] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const controllerRef = useRef(null);

  const loadProfile = useCallback(async ({ silent = false } = {}) => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;

    if (silent) {
      setRefreshing(true);
    } else {
      setLoading(true);
    }

    try {
      const response = await fetch(RESOURCES_ENDPOINT, {
        cache: "no-store",
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`Le serveur a répondu avec le statut ${response.status}.`);
      }

      const data = await response.json();
      setProfile(data);
      setError("");
    } catch (requestError) {
      if (requestError.name !== "AbortError") {
        setError(
          requestError.message || "Impossible de récupérer les ressources système.",
        );
      }
    } finally {
      if (!controller.signal.aborted) {
        setLoading(false);
        setRefreshing(false);
      }
    }
  }, []);

  useEffect(() => {
    const initialRequest = window.setTimeout(() => loadProfile(), 0);
    const interval = window.setInterval(
      () => loadProfile({ silent: true }),
      REFRESH_INTERVAL,
    );

    return () => {
      window.clearTimeout(initialRequest);
      window.clearInterval(interval);
      controllerRef.current?.abort();
    };
  }, [loadProfile]);

  const computed = useMemo(() => {
    if (!profile) return null;
    const snapshot = profile.system;
    const ram = snapshot?.ram ?? {};
    const storage = snapshot?.storage ?? {};
    const gpu = Array.isArray(snapshot?.gpus) ? snapshot.gpus[0] : null;
    const ramUsed = ram.used_bytes;
    const storageUsed = storage.used_bytes;
    const isProduction = profile.profile === "GPU_PRODUCTION";
    const hasNvidia = snapshot?.gpus?.some((item) =>
      `${item.manufacturer ?? ""} ${item.model ?? ""}`.toLowerCase().includes("nvidia"),
    );
    const coreMetricsKnown = Boolean(
      snapshot?.source === "host" &&
        snapshot?.cpu?.model &&
        ram.total_bytes &&
        storage.total_bytes,
    );
    const healthy = Boolean(
      coreMetricsKnown &&
        (!isProduction || (hasNvidia && profile.worker?.gpu)),
    );

    return {
      snapshot,
      gpu,
      healthy,
      ramUsed,
      ramPercent: getPercent(ramUsed, ram.total_bytes),
      storageUsed,
      storageVolumes:
        Array.isArray(storage.volumes) && storage.volumes.length > 0
          ? storage.volumes
          : storage.total_bytes
            ? [
              {
                name: "Système",
                mount_point: "/",
                filesystem: null,
                storage_type: "Agrégé",
                total_bytes: storage.total_bytes,
                used_bytes: storageUsed,
                available_bytes: storage.available_bytes,
              },
            ]
            : [],
      gpuPercent: gpu?.utilization_percent,
    };
  }, [profile]);

  return (
    <div className={styles.page}>
      <header className={styles.pageHeader}>
        <div>
          <h1>Ressources</h1>
          <p>Vue d’ensemble des ressources système de votre serveur.</p>
        </div>

        <div className={styles.headerActions}>
          <Button
            variant="secondary"
            icon={<BsArrowClockwise aria-hidden="true" />}
            loading={refreshing}
            onClick={() => loadProfile({ silent: true })}
          >
            Actualiser
          </Button>
          <span
            className={`${styles.liveStatus} ${error ? styles.liveStatusError : ""}`}
            role="status"
          >
            <span aria-hidden="true" />
            {error ? "Connexion interrompue" : "En temps réel"}
          </span>
        </div>
      </header>

      {loading && !profile ? (
        <section className={`${styles.panel} ${styles.statePanel}`}>
          <LoadingState
            title="Analyse du système…"
            description="Lecture du processeur, de la mémoire, du GPU et du stockage."
          />
        </section>
      ) : error && !profile ? (
        <section className={`${styles.panel} ${styles.statePanel}`}>
          <ErrorState
            title="Serveur indisponible"
            description={error}
            action={
              <Button
                variant="secondary"
                icon={<BsArrowClockwise aria-hidden="true" />}
                onClick={() => loadProfile()}
              >
                Réessayer
              </Button>
            }
          />
        </section>
      ) : (
        profile && computed && (
          <div className={styles.dashboard} aria-live="polite">
            {error && (
              <div className={styles.staleNotice} role="alert">
                Les dernières données connues sont affichées. {error}
              </div>
            )}

            {profile.system_error && (
              <div className={styles.staleNotice} role="status">
                {profile.system_error}
              </div>
            )}

            <section className={`${styles.panel} ${styles.statusPanel}`}>
              <div className={styles.healthSummary}>
                <span className={styles.serverIllustration} aria-hidden="true">
                  <BsServer />
                </span>
                <div>
                  <span>Statut du système</span>
                  <strong className={!computed.healthy ? styles.degradedStatus : ""}>
                    {computed.healthy ? (
                      <BsCheckCircleFill aria-hidden="true" />
                    ) : (
                      <BsExclamationTriangleFill aria-hidden="true" />
                    )}
                    {computed.healthy ? "Optimal" : "Données partielles"}
                  </strong>
                  <p>
                    Profil {profile.profile || "LOCAL"} · Source :{" "}
                    <span className={styles.sourceBadge}>
                      {computed.snapshot.source === "host" ? "Hôte" : "Conteneur"}
                    </span>
                  </p>
                </div>
              </div>

              <div className={styles.metricsGrid}>
                <Metric
                  label="Température GPU"
                  value={formatTemperature(computed.gpu?.temperature_celsius)}
                  icon={<BsThermometerHalf />}
                />
                <Metric
                  label="Charge système"
                  value={computed.snapshot.cpu.utilization_percent == null
                    ? "Non disponible"
                    : `${Math.round(computed.snapshot.cpu.utilization_percent)} %`}
                  icon={<BsActivity />}
                />
                <Metric
                  label="Uptime"
                  value={formatUptime(computed.snapshot.system.uptime_seconds)}
                  icon={<BsClockHistory />}
                />
              </div>
            </section>

            <div className={styles.summaryGrid}>
              <ResourcePanel
                title="Système"
                icon={<BsPcDisplayHorizontal />}
              >
                <div className={styles.infoList}>
                  <InfoRow
                    label="OS"
                    value={[
                      computed.snapshot.system.os,
                      computed.snapshot.system.os_version,
                    ].filter(Boolean).join(" ") || "Non disponible"}
                  />
                  <InfoRow
                    label="Kernel"
                    value={computed.snapshot.system.kernel || "Non disponible"}
                  />
                  <InfoRow
                    label="Architecture"
                    value={computed.snapshot.system.architecture || "Non disponible"}
                  />
                  <InfoRow
                    label="Hostname"
                    value={computed.snapshot.system.hostname || "Non disponible"}
                  />
                  <InfoRow
                    label="Date et heure"
                    value={formatDateTime(computed.snapshot.system.measured_at_unix)}
                  />
                </div>
              </ResourcePanel>

              <ResourcePanel title="Processeur (CPU)" icon={<BsCpu />}>
                <div className={styles.infoList}>
                  <InfoRow
                    label="Modèle"
                    value={computed.snapshot.cpu.model || "Non disponible"}
                  />
                  <InfoRow
                    label="Cœurs / Threads"
                    value={`${computed.snapshot.cpu.physical_cores ?? "—"} / ${computed.snapshot.cpu.logical_cpus ?? "—"}`}
                  />
                  <InfoRow
                    label="Fréquence"
                    value={formatFrequency(computed.snapshot.cpu.frequency_mhz)}
                  />
                  <InfoRow
                    label="Charge actuelle"
                    value={computed.snapshot.cpu.utilization_percent == null
                      ? "Non disponible"
                      : `${Math.round(computed.snapshot.cpu.utilization_percent)} %`}
                  >
                    {computed.snapshot.cpu.utilization_percent != null && (
                      <ProgressBar
                        value={computed.snapshot.cpu.utilization_percent}
                        size="sm"
                        className={styles.greenProgress}
                      />
                    )}
                  </InfoRow>
                  <InfoRow
                    label="Température"
                    value={formatTemperature(computed.snapshot.cpu.temperature_celsius)}
                  />
                </div>
              </ResourcePanel>

              <ResourcePanel title="Mémoire (RAM)" icon={<BsMemory />}>
                <div className={styles.memoryLayout}>
                  {computed.snapshot.ram.total_bytes == null ? (
                    <div className={styles.unknownDial}>Non disponible</div>
                  ) : (
                    <CircularProgress
                      value={computed.ramPercent}
                      size={126}
                      strokeWidth={13}
                      label="Utilisée"
                    />
                  )}
                  <div className={styles.memoryStats}>
                    <InfoRow
                      label="Total"
                      value={formatBytes(computed.snapshot.ram.total_bytes)}
                    />
                    <InfoRow label="Utilisée" value={formatBytes(computed.ramUsed)} />
                    <InfoRow
                      label="Libre"
                      value={formatBytes(computed.snapshot.ram.available_bytes)}
                    />
                    <InfoRow
                      label="Type"
                      value={computed.snapshot.ram.memory_type || "Non disponible"}
                    />
                  </div>
                </div>
              </ResourcePanel>
            </div>

            <ResourcePanel
              title="Carte graphique (GPU)"
              icon={<BsGpuCard />}
              className={styles.gpuPanel}
            >
              <div className={styles.gpuLayout}>
                <div className={styles.gpuIdentity}>
                  <span className={styles.gpuBadge} aria-hidden="true">
                    <BsGpuCard />
                  </span>
                  <div>
                    <strong>{computed.gpu?.model || "Aucun GPU physique détecté"}</strong>
                    <span>
                      {computed.gpu?.manufacturer || "Accélération non disponible"}
                      {computed.gpu?.backend ? ` · ${computed.gpu.backend}` : ""}
                    </span>
                  </div>
                </div>

                <div className={styles.gpuMetrics}>
                  <div>
                    <span>Mémoire (VRAM)</span>
                    <strong>
                      {computed.gpu?.backend === "Metal" && computed.gpu?.vram_total_bytes == null
                        ? "Mémoire unifiée"
                        : formatBytes(computed.gpu?.vram_total_bytes)}
                    </strong>
                  </div>
                  <div>
                    <span>Utilisée</span>
                    <strong>{formatBytes(computed.gpu?.vram_used_bytes)}</strong>
                  </div>
                  <div>
                    <span>Disponible</span>
                    <strong>{formatBytes(computed.gpu?.vram_available_bytes)}</strong>
                  </div>
                  <div className={styles.gpuUsage}>
                    <span>Utilisation</span>
                    <strong>
                      {computed.gpuPercent == null
                        ? "Non disponible"
                        : `${Math.round(computed.gpuPercent)} %`}
                    </strong>
                    {computed.gpuPercent != null && (
                      <ProgressBar
                        value={computed.gpuPercent}
                        size="sm"
                        className={styles.greenProgress}
                      />
                    )}
                  </div>
                  <div>
                    <span>Température</span>
                    <strong>{formatTemperature(computed.gpu?.temperature_celsius)}</strong>
                  </div>
                  <div>
                    <span>Driver</span>
                    <strong>{computed.gpu?.driver_version || "Non disponible"}</strong>
                  </div>
                  <div>
                    <span>CUDA</span>
                    <strong>
                      {computed.gpu?.backend === "CUDA"
                        ? computed.gpu.runtime_version || "Non disponible"
                        : "Non disponible"}
                    </strong>
                  </div>
                </div>
              </div>
            </ResourcePanel>

            <ResourcePanel
              title="Stockage"
              icon={<BsHddStack />}
              className={styles.storagePanel}
            >
              <div className={styles.storageHeader} aria-hidden="true">
                <span>Volume</span>
                <span>Type</span>
                <span>Total</span>
                <span>Utilisé</span>
                <span>Disponible</span>
                <span>Utilisation</span>
              </div>

              {computed.storageVolumes.map((volume) => {
                const volumePercent = getPercent(volume.used_bytes, volume.total_bytes);
                const type = [volume.storage_type, volume.filesystem?.toUpperCase()]
                  .filter(Boolean)
                  .join(" · ");
                const label = volume.mount_point === "/"
                  ? "(Système)"
                  : volume.mount_point || volume.name;

                return (
                  <div
                    className={styles.storageRow}
                    key={`${volume.name}-${volume.mount_point}`}
                  >
                    <div className={styles.storageVolume} title={volume.name || "Volume"}>
                      <BsDeviceSsd aria-hidden="true" />
                      <strong>{label}</strong>
                    </div>
                    <span>{type || "Non renseigné"}</span>
                    <strong>{formatBytes(volume.total_bytes)}</strong>
                    <strong>{formatBytes(volume.used_bytes)}</strong>
                    <strong>{formatBytes(volume.available_bytes)}</strong>
                    <div className={styles.storageUsage}>
                      <ProgressBar
                        value={volumePercent}
                        size="sm"
                        className={styles.greenProgress}
                      />
                      <strong>{Math.round(volumePercent)} %</strong>
                    </div>
                  </div>
                );
              })}

              <div className={styles.storageFooter}>
                <span>
                  Espace disque total :{" "}
                  <strong>{formatBytes(computed.snapshot.storage.total_bytes)}</strong>
                </span>
                <span>
                  Espace disponible :{" "}
                  <strong>{formatBytes(computed.snapshot.storage.available_bytes)}</strong>
                </span>
              </div>
            </ResourcePanel>
          </div>
        )
      )}
    </div>
  );
}
