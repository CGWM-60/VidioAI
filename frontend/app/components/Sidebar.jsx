"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BsBox,
  BsCardImage,
  BsChatDots,
  BsChevronDown,
  BsFolder,
  BsFolder2Open,
  BsGear,
  BsGrid1X2,
  BsHouse,
  BsQuestionCircle,
  BsRobot,
  BsStars,
} from "react-icons/bs";

const navigation = [
  { href: "/", label: "Accueil", icon: BsHouse },
  { href: "/models", label: "Modèles", icon: BsBox },
  { href: "/projects", label: "Projets", icon: BsFolder },
  { href: "/generations", label: "Générations", icon: BsRobot },
  { href: "/images", label: "Images", icon: BsCardImage },
  { href: "/chat", label: "Chat", icon: BsChatDots },
  { href: "/resources", label: "Ressources", icon: BsFolder2Open },
  { href: "/settings", label: "Paramètres", icon: BsGear },
  { href: "/ui-demo", label: "Interface UI", icon: BsGrid1X2 },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="sidebar">
      <Link className="sidebar-brand" href="/" aria-label="VidioAI — Accueil">
        <span className="sidebar-brand-mark"><BsStars /></span>
        <span>VidioAI</span>
      </Link>

      <nav className="sidebar-nav" aria-label="Navigation principale">
        {navigation.map(({ href, label, icon: Icon }) => {
          const active = href === "/"
            ? pathname === href
            : pathname === href || pathname.startsWith(`${href}/`);

          return (
            <Link
              href={href}
              key={href}
              className={`sidebar-link ${active ? "active" : ""}`}
              aria-current={active ? "page" : undefined}
            >
              <Icon aria-hidden="true" />
              <span>{label}</span>
            </Link>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        <Link href="#" className="sidebar-help">
          <BsQuestionCircle aria-hidden="true" />
          <span>Aide</span>
        </Link>

        <button className="sidebar-profile" type="button">
          <span className="sidebar-avatar" aria-hidden="true">A</span>
          <span className="sidebar-profile-copy">
            <strong>Alex</strong>
            <small>Plan Pro</small>
          </span>
          <BsChevronDown className="sidebar-profile-chevron" aria-hidden="true" />
        </button>
      </div>
    </aside>
  );
}
