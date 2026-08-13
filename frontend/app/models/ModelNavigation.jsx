import Link from "next/link";
import styles from "../studio.module.css";

const ITEMS = [
  ["catalog", "/models", "Catalogue"],
  ["installed", "/models/installed", "Installés"],
  ["cloud", "/models/cloud", "Cloud"],
  ["lab", "/models/lab", "Lab"],
];

export default function ModelNavigation({ active }) {
  return (
    <nav className={styles.modelNavigation} aria-label="Navigation modèles">
      {ITEMS.map(([key, href, label]) => (
        <Link
          className={active === key ? styles.modelNavigationActive : undefined}
          aria-current={active === key ? "page" : undefined}
          href={href}
          key={key}
        >
          {label}
        </Link>
      ))}
    </nav>
  );
}
