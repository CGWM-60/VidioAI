"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BsBox,
  BsBeaker,
  BsCardImage,
  BsChatDots,
  BsCloudArrowDown,
  BsChevronDown,
  BsFolder,
  BsFolder2Open,
  BsGear,
  BsGrid1X2,
  BsHouse,
  BsHddStack,
  BsQuestionCircle,
  BsRobot,
  BsStars,
} from "react-icons/bs";

const navigation = [
  { href: "/", label: "Accueil", icon: BsHouse },
  { href: "/models", label: "Modèles", icon: BsBox },
  { href: "/models/installed", label: "Modèles installés", icon: BsHddStack },
  { href: "/models/cloud", label: "Sauvegardes cloud", icon: BsCloudArrowDown },
  { href: "/models/lab", label: "VidioAI Lab", icon: BsBeaker },
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
  const version = process.env.NEXT_PUBLIC_VIDIOAI_VERSION;
  const activeHref = navigation
    .filter(({ href }) => href === "/" ? pathname === "/" : pathname === href || pathname.startsWith(`${href}/`))
    .sort((left, right) => right.href.length - left.href.length)[0]?.href;

  return (
    <aside className="sidebar">
      <Link className="sidebar-brand" href="/" aria-label="VidioAI — Accueil">
        <span className="sidebar-brand-mark"><BsStars /></span>
        <span className="sidebar-brand-copy"><span>VidioAI</span>{version && <small>v{version}</small>}</span>
      </Link>

      <nav className="sidebar-nav" aria-label="Navigation principale">
        {navigation.map(({ href, label, icon: Icon }) => {
          const active = href === activeHref;

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
