"use client";

import { useState } from "react";
import {
  BsArrowRight,
  BsCheck2,
  BsClipboard,
  BsCloudArrowUp,
  BsCodeSlash,
  BsDownload,
  BsExclamationTriangle,
  BsLightningCharge,
  BsPalette,
  BsSliders,
  BsStars,
  BsTrash3,
  BsX,
} from "react-icons/bs";
import {
  Badge,
  Button,
  Card,
  CircularProgress,
  EmptyState,
  ErrorState,
  Input,
  LoadingState,
  Modal,
  Panel,
  ProgressBar,
  Select,
  Tabs,
  Textarea,
  Toast,
  Toggle,
  UploadZone,
} from "../components/ui";

const qualityTabs = [
  { id: "fast", label: "Rapide", subtitle: "720p" },
  {
    id: "balanced",
    label: "Équilibrée",
    subtitle: "1080p",
    showCheck: true,
  },
  { id: "max", label: "Qualité max", subtitle: "2K" },
];

const examples = {
  button: {
    name: "Button",
    importCode: `import { BsDownload } from "react-icons/bs";
import { Button } from "@/app/components/ui";`,
    implementationCode: `<Button icon={<BsDownload />}>Installer</Button>
<Button variant="secondary">Annuler</Button>
<Button variant="danger">Supprimer</Button>
<Button variant="ghost">Détails</Button>
<Button loading>Chargement</Button>`,
  },
  badge: {
    name: "Badge",
    importCode: 'import { Badge } from "@/app/components/ui";',
    implementationCode: `<Badge>Texte → Vidéo</Badge>
<Badge variant="green" dot>Prêt</Badge>
<Badge variant="warning" dot>En attente</Badge>
<Badge variant="info">1080p</Badge>`,
  },
  card: {
    name: "Card",
    importCode: 'import { Card, Badge } from "@/app/components/ui";',
    implementationCode: `<Card interactive selected>
  <h3>MiniMax H3</h3>
  <p>Génération vidéo réaliste.</p>
  <Badge variant="green" dot>Prêt</Badge>
</Card>`,
  },
  input: {
    name: "Input",
    importCode: 'import { Input } from "@/app/components/ui";',
    implementationCode: `<Input
  label="Nom du modèle"
  placeholder="MiniMax H3"
  help="Utilisez un nom facile à retrouver."
/>`,
  },
  select: {
    name: "Select",
    importCode: 'import { Select } from "@/app/components/ui";',
    implementationCode: `<Select
  label="Qualité"
  defaultValue="1080"
  options={[
    { value: "720", label: "Rapide (720p)" },
    { value: "1080", label: "Équilibrée (1080p)" },
    { value: "2k", label: "Qualité max (2K)" },
  ]}
/>`,
  },
  textarea: {
    name: "Textarea",
    importCode: `import { useState } from "react";
import { Textarea } from "@/app/components/ui";`,
    implementationCode: `const [prompt, setPrompt] = useState("");

<Textarea
  label="Prompt"
  value={prompt}
  onChange={(event) => setPrompt(event.target.value)}
  maxLength={1000}
  placeholder="Décrivez votre vidéo…"
/>`,
  },
  tabs: {
    name: "Tabs",
    importCode: `import { useState } from "react";
import { Tabs } from "@/app/components/ui";`,
    implementationCode: `const [activeTab, setActiveTab] = useState("balanced");

<Tabs
  fluid
  active={activeTab}
  onChange={setActiveTab}
  tabs={[
    { id: "fast", label: "Rapide", subtitle: "720p" },
    { id: "balanced", label: "Équilibrée", subtitle: "1080p" },
  ]}
/>`,
  },
  toggle: {
    name: "Toggle",
    importCode: `import { useState } from "react";
import { Toggle } from "@/app/components/ui";`,
    implementationCode: `const [enabled, setEnabled] = useState(true);

<Toggle
  checked={enabled}
  onChange={setEnabled}
  label="Optimisation automatique"
  description="Ajuste les ressources pour de meilleurs résultats."
/>`,
  },
  progress: {
    name: "ProgressBar",
    importCode: 'import { ProgressBar } from "@/app/components/ui";',
    implementationCode: `<ProgressBar
  value={75}
  label="Installation en cours…"
  showValue
  size="lg"
/>`,
  },
  circularProgress: {
    name: "CircularProgress",
    importCode: 'import { CircularProgress } from "@/app/components/ui";',
    implementationCode: `<CircularProgress
  value={75}
  size={180}
  strokeWidth={12}
  label="Installation en cours…"
  sublabel="Cela peut prendre quelques minutes."
/>`,
  },
  upload: {
    name: "UploadZone",
    importCode: 'import { UploadZone } from "@/app/components/ui";',
    implementationCode: `function handleFiles(files) {
  // Envoyez ou prévisualisez les fichiers sélectionnés.
}

<UploadZone
  accept="image/png,image/jpeg,image/webp"
  title="Glissez-déposez une image ici"
  description="ou cliquez pour parcourir"
  helper="JPG, PNG, WebP — 15 Mo max"
  onFiles={handleFiles}
/>`,
  },
  empty: {
    name: "EmptyState",
    importCode: 'import { EmptyState } from "@/app/components/ui";',
    implementationCode: `<EmptyState
  compact
  title="Aucun élément"
  description="Il n’y a encore rien à afficher."
/>`,
  },
  error: {
    name: "ErrorState",
    importCode: 'import { ErrorState } from "@/app/components/ui";',
    implementationCode: `<ErrorState
  compact
  title="Une erreur est survenue"
  description="Impossible de terminer l’opération."
/>`,
  },
  loading: {
    name: "LoadingState",
    importCode: 'import { LoadingState } from "@/app/components/ui";',
    implementationCode: `<LoadingState
  compact
  title="Chargement…"
/>`,
  },
  modal: {
    name: "Modal",
    importCode: `import { useState } from "react";
import { Modal, Button } from "@/app/components/ui";`,
    implementationCode: `const [modalOpen, setModalOpen] = useState(false);

<>
  <Button onClick={() => setModalOpen(true)}>Ouvrir</Button>
  <Modal
    open={modalOpen}
    onClose={() => setModalOpen(false)}
    title="Installer MiniMax H3"
    subtitle="L’installation sera faite automatiquement."
    footer={<Button>Installer</Button>}
  >
    Contenu de la fenêtre
  </Modal>
</>`,
  },
  toast: {
    name: "Toast",
    importCode: `import { useState } from "react";
import { Toast, Button } from "@/app/components/ui";`,
    implementationCode: `const [toastOpen, setToastOpen] = useState(false);

<>
  <Button onClick={() => setToastOpen(true)}>Notifier</Button>
  <Toast
    open={toastOpen}
    onClose={() => setToastOpen(false)}
    title="Installation terminée"
    variant="success"
  >
    MiniMax H3 est prêt à être utilisé.
  </Toast>
</>`,
  },
};

function ExampleTarget({ id, active, onSelect, children, className = "" }) {
  return (
    <div
      className={`ui-demo-example ${active === id ? "ui-demo-example-active" : ""} ${className}`}
      data-example={examples[id].name}
      onClickCapture={() => onSelect(id)}
    >
      {children}
    </div>
  );
}

function CodeExample({ id }) {
  const [copied, setCopied] = useState(false);
  const example = examples[id];

  async function copyCode() {
    const code = `${example.importCode}\n\n${example.implementationCode}`;

    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(code);
    } else {
      const textarea = document.createElement("textarea");
      textarea.value = code;
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
    }

    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  }

  return (
    <div className="ui-demo-code-example" aria-live="polite">
      <div className="ui-demo-code-header">
        <div>
          <BsCodeSlash aria-hidden="true" />
          <span>Exemple · {example.name}</span>
        </div>
        <button type="button" onClick={copyCode} className="ui-demo-copy-button">
          {copied ? <BsCheck2 /> : <BsClipboard />}
          <span>{copied ? "Copié" : "Copier"}</span>
        </button>
      </div>

      <div className="ui-demo-code-section">
        <span className="ui-demo-code-label">Import</span>
        <pre><code>{example.importCode}</code></pre>
      </div>
      <div className="ui-demo-code-section">
        <span className="ui-demo-code-label">Implémentation</span>
        <pre><code>{example.implementationCode}</code></pre>
      </div>
    </div>
  );
}

function PanelCode({ active, ids }) {
  return ids.includes(active) ? <CodeExample id={active} /> : null;
}

export default function UiDemoPage() {
  const [toggle, setToggle] = useState(true);
  const [tab, setTab] = useState("balanced");
  const [prompt, setPrompt] = useState(
    "Une voiture volante traverse une ville futuriste au coucher du soleil.",
  );
  const [modalOpen, setModalOpen] = useState(false);
  const [toastOpen, setToastOpen] = useState(false);
  const [activeExample, setActiveExample] = useState(null);

  return (
    <div className="ui-demo-page">
      <header className="ui-demo-header">
        <div>
          <div className="ui-demo-eyebrow">
            <BsPalette aria-hidden="true" />
            Design system
          </div>
          <h1>Bibliothèque d’interface</h1>
          <p>
            Les composants de VidioAI, alignés sur l’interface sombre et
            violette de l’application.
          </p>
          <div className="ui-demo-code-hint">
            <BsCodeSlash aria-hidden="true" />
            Cliquez sur un composant pour afficher son code.
          </div>
        </div>

        <div className="ui-demo-header-actions">
          <Button variant="secondary" icon={<BsSliders />}>
            Personnaliser
          </Button>
          <Button icon={<BsStars />}>Nouveau projet</Button>
        </div>
      </header>

      <section className="ui-demo-overview" aria-label="Aperçu du système UI">
        <Card className="ui-demo-stat" padding={false}>
          <span className="ui-demo-stat-icon">
            <BsLightningCharge />
          </span>
          <div>
            <strong>17</strong>
            <span>composants</span>
          </div>
        </Card>
        <Card className="ui-demo-stat" padding={false}>
          <span className="ui-demo-stat-icon">
            <BsPalette />
          </span>
          <div>
            <strong>6</strong>
            <span>couleurs d’état</span>
          </div>
        </Card>
        <Card className="ui-demo-stat" padding={false}>
          <span className="ui-demo-stat-icon ui-demo-stat-icon-success">
            <BsCheck2 />
          </span>
          <div>
            <strong>100%</strong>
            <span>responsive</span>
          </div>
        </Card>
      </section>

      <div className="ui-demo-grid">
        <Panel
          className="ui-demo-span-7"
          title="Actions"
          subtitle="Boutons principaux, secondaires et contextuels."
          icon={<BsLightningCharge />}
          actions={<Badge variant="green" dot>Prêt</Badge>}
        >
          <ExampleTarget id="button" active={activeExample} onSelect={setActiveExample}>
            <div className="ui-demo-button-row">
              <Button icon={<BsDownload />}>Installer</Button>
              <Button variant="secondary" icon={<BsX />}>Annuler</Button>
              <Button variant="danger" icon={<BsTrash3 />}>Supprimer</Button>
              <Button variant="ghost" icon={<BsArrowRight />}>Détails</Button>
              <Button loading>Chargement</Button>
            </div>
          </ExampleTarget>

          <ExampleTarget id="badge" active={activeExample} onSelect={setActiveExample}>
            <div className="ui-demo-badges" aria-label="Badges d’état">
              <Badge>Texte → Vidéo</Badge>
              <Badge variant="green" dot>Prêt</Badge>
              <Badge variant="warning" dot>En attente</Badge>
              <Badge variant="info">1080p</Badge>
              <Badge variant="red">Erreur</Badge>
              <Badge variant="gray">Archivé</Badge>
            </div>
          </ExampleTarget>
          <PanelCode active={activeExample} ids={["button", "badge"]} />
        </Panel>

        <Panel
          className="ui-demo-span-5"
          title="Modèle installé"
          subtitle="Exemple de carte interactive."
          icon={<BsStars />}
        >
          <ExampleTarget id="card" active={activeExample} onSelect={setActiveExample}>
            <Card interactive selected className="ui-demo-model-card">
              <div className="ui-demo-model-art" aria-hidden="true">
                <BsStars />
              </div>
              <div className="ui-demo-model-copy">
                <div className="ui-demo-model-heading">
                  <h3>MiniMax H3</h3>
                  <Badge variant="green" dot>Prêt</Badge>
                </div>
                <p>Génération vidéo réaliste à haute cohérence temporelle.</p>
                <div className="ui-demo-badges">
                  <Badge>Image → Vidéo</Badge>
                  <Badge variant="gray">v1.0</Badge>
                </div>
              </div>
            </Card>
          </ExampleTarget>
          <PanelCode active={activeExample} ids={["card"]} />
        </Panel>

        <Panel
          className="ui-demo-span-7"
          title="Formulaires"
          subtitle="Champs, listes et zones de texte."
          icon={<BsSliders />}
        >
          <div className="ui-demo-form-grid">
            <ExampleTarget id="input" active={activeExample} onSelect={setActiveExample}>
              <Input
                label="Nom du modèle"
                placeholder="MiniMax H3"
                help="Utilisez un nom facile à retrouver."
              />
            </ExampleTarget>
            <ExampleTarget id="select" active={activeExample} onSelect={setActiveExample}>
              <Select
                label="Qualité"
                defaultValue="1080"
                options={[
                  { value: "720", label: "Rapide (720p)" },
                  { value: "1080", label: "Équilibrée (1080p)" },
                  { value: "2k", label: "Qualité max (2K)" },
                ]}
              />
            </ExampleTarget>
            <ExampleTarget
              id="textarea"
              active={activeExample}
              onSelect={setActiveExample}
              className="ui-demo-form-wide"
            >
              <Textarea
                label="Prompt"
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                maxLength={1000}
                placeholder="Décrivez votre vidéo…"
              />
            </ExampleTarget>
          </div>
          <PanelCode active={activeExample} ids={["input", "select", "textarea"]} />
        </Panel>

        <Panel
          className="ui-demo-span-5"
          title="Préférences"
          subtitle="Sélections segmentées et interrupteurs."
          icon={<BsSliders />}
        >
          <div className="ui-demo-stack">
            <ExampleTarget id="tabs" active={activeExample} onSelect={setActiveExample}>
              <span className="ui-demo-field-title">Qualité de rendu</span>
              <Tabs
                fluid
                active={tab}
                onChange={setTab}
                tabs={qualityTabs}
              />
            </ExampleTarget>
            <ExampleTarget id="toggle" active={activeExample} onSelect={setActiveExample}>
              <div className="ui-demo-toggle-list">
                <Toggle
                  checked={toggle}
                  onChange={setToggle}
                  label="Optimisation automatique"
                  description="Ajuste les ressources pour de meilleurs résultats."
                />
                <Toggle
                  checked
                  onChange={() => {}}
                  label="Utiliser toute la VRAM"
                  description="Maximise les performances de génération."
                />
              </div>
            </ExampleTarget>
          </div>
          <PanelCode active={activeExample} ids={["tabs", "toggle"]} />
        </Panel>

        <Panel
          className="ui-demo-span-12"
          title="Progression"
          subtitle="Suivi d’installation et de génération en temps réel."
          icon={<BsLightningCharge />}
        >
          <div className="ui-demo-progress-layout">
            <ExampleTarget
              id="progress"
              active={activeExample}
              onSelect={setActiveExample}
              className="ui-demo-progress-copy"
            >
              <Badge variant="info">Installation automatique</Badge>
              <h3>Installation de MiniMax H3</h3>
              <p>Extraction et optimisation des fichiers du modèle.</p>
              <ProgressBar
                value={75}
                label="Installation en cours…"
                showValue
                size="lg"
              />
              <span className="ui-demo-progress-meta">
                Temps restant estimé : 00:25
              </span>
            </ExampleTarget>
            <ExampleTarget
              id="circularProgress"
              active={activeExample}
              onSelect={setActiveExample}
              className="ui-demo-circular-card"
            >
              <CircularProgress
                value={75}
                size={180}
                strokeWidth={12}
                label="Installation en cours…"
                sublabel="Cela peut prendre quelques minutes."
              />
            </ExampleTarget>
          </div>
          <PanelCode active={activeExample} ids={["progress", "circularProgress"]} />
        </Panel>

        <Panel
          className="ui-demo-span-6"
          title="Import de fichier"
          subtitle="Zone de dépôt accessible au clavier."
          icon={<BsCloudArrowUp />}
        >
          <ExampleTarget id="upload" active={activeExample} onSelect={setActiveExample}>
            <UploadZone
              accept="image/png,image/jpeg,image/webp"
              title="Glissez-déposez une image ici"
              description="ou cliquez pour parcourir"
              helper="JPG, PNG, WebP — 15 Mo max"
              icon={<BsCloudArrowUp />}
              onFiles={() => {}}
            />
          </ExampleTarget>
          <PanelCode active={activeExample} ids={["upload"]} />
        </Panel>

        <Panel
          className="ui-demo-span-6"
          title="États système"
          subtitle="Vide, erreur et chargement."
          icon={<BsExclamationTriangle />}
        >
          <div className="ui-demo-states">
            <ExampleTarget id="empty" active={activeExample} onSelect={setActiveExample}>
              <Card padding={false}><EmptyState compact /></Card>
            </ExampleTarget>
            <ExampleTarget id="error" active={activeExample} onSelect={setActiveExample}>
              <Card padding={false}><ErrorState compact /></Card>
            </ExampleTarget>
            <ExampleTarget id="loading" active={activeExample} onSelect={setActiveExample}>
              <Card padding={false}><LoadingState compact /></Card>
            </ExampleTarget>
          </div>
          <PanelCode active={activeExample} ids={["empty", "error", "loading"]} />
        </Panel>

        <Panel
          className="ui-demo-span-12"
          title="Calques et notifications"
          subtitle="Modal de confirmation et toast de succès."
          icon={<BsStars />}
        >
          <div className="ui-demo-button-row">
            <Button onClick={() => {
              setActiveExample("modal");
              setModalOpen(true);
            }}>
              Ouvrir la modal
            </Button>
            <Button variant="secondary" onClick={() => {
              setActiveExample("toast");
              setToastOpen(true);
            }}>
              Afficher le toast
            </Button>
          </div>
          <PanelCode active={activeExample} ids={["modal", "toast"]} />
        </Panel>
      </div>

      <Modal
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        title="Installer MiniMax H3"
        subtitle="L’installation sera faite automatiquement."
        footer={
          <>
            <Button variant="secondary" onClick={() => setModalOpen(false)}>
              Annuler
            </Button>
            <Button onClick={() => setModalOpen(false)}>Installer</Button>
          </>
        }
      >
        <div className="ui-demo-modal-summary">
          <span><BsDownload /></span>
          <div>
            <strong>64,2 Go à télécharger</strong>
            <p>Le modèle sera vérifié avant son installation.</p>
          </div>
        </div>
      </Modal>

      <Toast
        open={toastOpen}
        onClose={() => setToastOpen(false)}
        title="Installation terminée"
        variant="success"
      >
        MiniMax H3 est prêt à être utilisé.
      </Toast>
    </div>
  );
}
