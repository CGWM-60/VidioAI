"use client";

import { useState } from "react";

export default function ChatInput({
    setMessages,
    socketRef,
    connected
}) {

    const [input, setInput] = useState("");


    // ============================
    // ENVOYER MESSAGE
    // ============================

    function sendMessage() {

        const text = input.trim();

        if (!text) {
            return;
        }


        // Vérification WebSocket

        // La ref est lue dans le gestionnaire d'événement, jamais pendant le
        // rendu React. Elle contient ainsi toujours la connexion la plus récente.
        const socket = socketRef.current;

        if (
            !socket ||
            socket.readyState !== WebSocket.OPEN
        ) {

            console.log(
                "WebSocket non connecté"
            );

            return;
        }


        // Message utilisateur

        const newMessage = {

            id: Date.now(),

            role: "user",

            content: text,

        };


        // Ajout dans l'interface

        setMessages((oldMessages) => [

            ...oldMessages,

            newMessage

        ]);


        // Envoi vers Rust

        socket.send(text);


        // Vider le textarea

        setInput("");

    }


    return (

        <div className="chat-input border-top bg-white px-4 py-3">

            <div className="chat-width mx-auto">


                <div className="chat-input-box border bg-white shadow-sm">


                    {/* TEXTAREA */}

                    <textarea

                        value={input}

                        onChange={(event) => {

                            setInput(
                                event.target.value
                            );

                        }}

                        onKeyDown={(event) => {

                            if (
                                event.key === "Enter" &&
                                !event.shiftKey
                            ) {

                                event.preventDefault();

                                sendMessage();

                            }

                        }}

                        placeholder="Envoyer un message..."

                        rows="1"

                        className="chat-textarea form-control border-0 shadow-none"

                    />


                    {/* ========================= */}
                    {/* ACTIONS */}
                    {/* ========================= */}

                    <div className="d-flex align-items-center justify-content-between px-3 pb-3">


                        {/* GAUCHE */}

                        <div className="d-flex align-items-center gap-2">

                            <button
                                type="button"
                                className="btn btn-light rounded-circle"
                            >
                                +
                            </button>


                            <button
                                type="button"
                                className="btn btn-light rounded-pill"
                            >
                                🌐 Web
                            </button>

                        </div>


                        {/* ENVOYER */}

                        <button

                            type="button"

                            onClick={sendMessage}

                            disabled={
                                !input.trim() ||
                                !connected
                            }

                            className="send-button btn btn-dark d-flex align-items-center justify-content-center"

                        >

                            ↑

                        </button>

                    </div>

                </div>


                <p className="text-center text-secondary small mt-2 mb-0">
                    L&apos;IA peut faire des erreurs.
                </p>

            </div>

        </div>

    );

}
