"use client";

import { usePathname } from "next/navigation";

export default function Headers({connected}) {
    const pathname = usePathname();

    if (pathname === "/resources") {
        return null;
    }

    return (
        <>
            <header className="ui-demo-header" style={{ padding: "38px 34px 64px" }}>
                <div>
                    {connected ? (

                        <span className="small text-success">
                            ● Serveur OK
                        </span>

                    ) : (

                        <span className="small text-danger">
                            ● Probleme connection serveur
                        </span>

                    )}
                    <h1>Bonjour !! Bienvenue</h1>
                    <p>
                        Les composants de VidioAI, alignés sur l’interface sombre et
                        violette de l’application.
                    </p>

                </div>

            </header>
        </>
    )
}
