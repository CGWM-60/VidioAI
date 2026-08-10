import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import React, { useEffect, useRef } from 'react';

export default function Message({ messages }) {

    const bottomRef = useRef(null);

    useEffect(() => {
        // Défilement automatique fluide à chaque nouveau message
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]); 
    return (

        <main className="chat-messages flex-grow-1 overflow-auto">

            <div className="chat-width mx-auto px-3 py-4">


                {messages.map((message) => (

                    <div

                        key={message.id}

                        className={`
                            d-flex
                            mb-4
                            ${message.role === "user"
                                ? "justify-content-end"
                                : "justify-content-start"
                            }
                        `}

                    >


                        {message.role === "assistant" ? (

                            // =========================
                            // ASSISTANT
                            // =========================

                            <div className="assistant-message d-flex gap-3">


                                {/* ICONE IA */}

                                <div className="ai-avatar bg-dark text-white d-flex align-items-center justify-content-center fw-bold">

                                    AI

                                </div>


                                {/* MESSAGE */}

                                <div className="flex-grow-1">


                                    <div className="small fw-semibold text-secondary mb-2">

                                        Assistant

                                    </div>


                                    <div className="message-content">

                                        <ReactMarkdown
                                            rehypePlugins={[
                                                rehypeHighlight
                                            ]}
                                        >
                                            {message.content}
                                        </ReactMarkdown>

                                    </div>
                                        <div ref={bottomRef} />


                                    {/* ACTIONS */}

                                    <div className="d-flex gap-1 mt-3">


                                        <button
                                            type="button"
                                            className="btn btn-sm btn-light"
                                        >
                                            👍
                                        </button>


                                        <button
                                            type="button"
                                            className="btn btn-sm btn-light"
                                        >
                                            👎
                                        </button>


                                        <button
                                            type="button"
                                            className="btn btn-sm btn-light"
                                        >
                                            📋
                                        </button>


                                    </div>

                                </div>

                            </div>

                        ) : (

                            // =========================
                            // UTILISATEUR
                            // =========================

                            <div className="user-message">

                                {message.content}

                            </div>

                        )}

                    </div>

                ))}

            </div>

        </main>

    );

}