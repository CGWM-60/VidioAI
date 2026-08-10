"use client";

import { useEffect, useRef, useState } from "react";

import Message from "../components/Message";
import ChatInput from "../components/ChatInput";
import { chatEventsUrl } from "../lib/api";

export default function Chat() {

    const [messages, setMessages] = useState([
        {
            // L'identifiant d'accueil est constant : le rendu reste pur et
            // identique entre le serveur et l'hydratation du navigateur.
            id: "welcome-message",
            role: "assistant",
            content: "Bonjour 👋 Comment puis-je vous aider ?",
        },
    ]);

    const [connected, setConnected] = useState(false);

    const socketRef = useRef(null);


    useEffect(() => {

        let disposed = false;

        // Le flux utilise la même origine que l'application : aucun localhost
        // codé en dur ne casse le chat derrière Nginx ou en HTTPS.
        const socket = new WebSocket(chatEventsUrl());

        socketRef.current = socket;


        // =========================
        // CONNEXION
        // =========================

        socket.addEventListener("open", () => {

            if (disposed) {
                socket.close();
                return;
            }

            console.log("✅ WebSocket connecté");

            setConnected(true);

        });


        // =========================
        // MESSAGE RUST
        // =========================

        socket.addEventListener("message", (event) => {

            if (disposed) {
                return;
            }

            const token = event.data;

            if (!token) {
                return;
            }


            setMessages((oldMessages) => {

                const messagesCopy = [
                    ...oldMessages
                ];

                const lastMessage =
                    messagesCopy[
                        messagesCopy.length - 1
                    ];


                // Le message IA existe déjà :
                // on ajoute le token

                if (
                    lastMessage &&
                    lastMessage.role === "assistant"
                ) {

                    messagesCopy[
                        messagesCopy.length - 1
                    ] = {

                        ...lastMessage,

                        content:
                            lastMessage.content +
                            token,

                    };

                    return messagesCopy;

                }


                // Premier token de la réponse IA

                return [
                    ...messagesCopy,

                    {
                        id: Date.now(),
                        role: "assistant",
                        content: token,
                    },
                ];

            });

        });


        // =========================
        // ERREUR
        // =========================

        socket.addEventListener("error", () => {

            // Si React est simplement
            // en train de démonter le
            // premier useEffect en dev,
            // on ne considère pas cela
            // comme une vraie erreur.

            if (disposed) {
                return;
            }

            console.error(
                "❌ Véritable erreur WebSocket"
            );

            console.log(
                "readyState :",
                socket.readyState
            );

        });


        // =========================
        // FERMETURE
        // =========================

        socket.addEventListener("close", (event) => {

            if (disposed) {
                return;
            }

            console.log(
                "🔴 WebSocket fermé"
            );

            console.log(
                "Code :",
                event.code
            );

            console.log(
                "Raison :",
                event.reason
            );

            setConnected(false);

        });


        // =========================
        // CLEANUP REACT
        // =========================

        return () => {

            disposed = true;

            if (
                socket.readyState === WebSocket.OPEN
            ) {
                socket.close();
            }

            if (
                socketRef.current === socket
            ) {
                socketRef.current = null;
            }

        };

    }, []);


    return (

        <div className="chat-page d-flex flex-column">


            {/* HEADER */}

            <header
                className="
                    chat-header
                    d-flex
                    align-items-center
                    justify-content-between
                    border-bottom
                    px-4
                "
            >

                <div>

                    <h1 className="fs-6 fw-semibold mb-1">
                        Qwen 2.5 0.5B
                    </h1>


                    {connected ? (

                        <span className="small text-success">
                            ● Modèle local connecté
                        </span>

                    ) : (

                        <span className="small text-danger">
                            ● Connexion...
                        </span>

                    )}

                </div>


                <button
                    type="button"
                    className="btn btn-light"
                >
                    ⚙️
                </button>

            </header>


            {/* MESSAGES */}

            <Message
                messages={messages}
            />


            {/* INPUT */}

            <ChatInput
                setMessages={setMessages}
                socketRef={socketRef}
                connected={connected}
            />


        </div>

    );

}
