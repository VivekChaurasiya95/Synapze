/**
 * AI Chat Component
 * Main chat interface for AI interactions
 */

import { useState, useEffect, useRef } from "react";
import { Loading, Alert } from "./index.js";
import {
  sendMessage,
  getChatHistory,
  clearChatHistory,
} from "../services/aiService.js";
import { useAuth } from "../context/index.js";
import "../styles/aiChat.css";

export const AIChat = ({ taskId, submissionId, onClose }) => {
  const { user } = useAuth();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [hint, setHint] = useState("");
  const messagesEndRef = useRef(null);

  // Load conversation history on mount
  useEffect(() => {
    loadHistory();
  }, []);

  // Auto-scroll to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  /**
   * Load conversation history
   */
  const loadHistory = async () => {
    try {
      const result = await getChatHistory();
      if (result.success && result.history) {
        setMessages(result.history);
      }
    } catch (err) {
      console.error("Error loading history:", err);
    }
  };

  /**
   * Update hint based on input
   */
  useEffect(() => {
    const triggers = {
      help: "Ask about concepts or approach for this task",
      submit: "Get feedback on your submission",
      mentor: "Request mentorship guidance",
      explain: "Get explanation of concepts",
      resources: "Find learning resources",
      recommend: "Get task recommendations",
      debug: "Debug or understand code",
    };

    for (const [trigger, hintText] of Object.entries(triggers)) {
      if (input.toLowerCase().includes(trigger)) {
        setHint(hintText);
        return;
      }
    }
    setHint("");
  }, [input]);

  /**
   * Handle sending message
   */
  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    // Check authentication
    if (!user) {
      setError("You must be logged in to use AI assistant");
      return;
    }

    const userMessage = input.trim();
    if (userMessage.length < 3) {
      setError("Please type at least 3 characters before sending.");
      return;
    }

    setInput("");
    setError("");
    setLoading(true);

    // Add user message to display
    const newMessage = {
      sender: "user",
      content: userMessage,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, newMessage]);

    try {
      const result = await sendMessage({
        message: userMessage,
        taskId,
        submissionId,
        conversationHistory: messages,
      });

      if (result && result.success) {
        setMessages((prev) => [
          ...prev,
          {
            sender: "assistant",
            content: result.response,
            metadata: result.metadata,
            timestamp: new Date(),
          },
        ]);
      } else {
        const errorMessage =
          result?.error || result?.message || result?.errors?.[0] || "Failed to get response";
        setError(errorMessage);
        setMessages((prev) => [
          ...prev,
          {
            sender: "system",
            content: `Error: ${errorMessage}`,
            timestamp: new Date(),
          },
        ]);
      }
    } catch (err) {
      console.error("Error in handleSendMessage:", err);
      let errorMsg = "An error occurred processing your request";
      if (err?.response?.data?.error) {
        errorMsg = err.response.data.error;
      } else if (err?.message) {
        errorMsg = err.message;
      } else if (err?.error) {
        errorMsg = err.error;
      } else if (typeof err === "string") {
        errorMsg = err;
      }
      setError(errorMsg);
      setMessages((prev) => [
        ...prev,
        {
          sender: "system",
          content: `Error: ${errorMsg}`,
          timestamp: new Date(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Handle clear history
   */
  const handleClearHistory = async () => {
    if (window.confirm("Clear conversation history?")) {
      try {
        await clearChatHistory();
        setMessages([]);
      } catch (err) {
        setError("Failed to clear history");
      }
    }
  };

  return (
    <div className="ai-chat-container">
      <div className="ai-chat-header">
        <div className="ai-header-content">
          <h3>Buddy AI</h3>
          <span className="ai-model-badge">Powered by Buddy</span>
        </div>
        <div className="ai-chat-actions">
          <span className="ai-role-badge">{user?.role || "Student"}</span>
          <button
            onClick={handleClearHistory}
            className="ai-action-btn"
            title="Clear history"
          >
            Clear
          </button>
          {onClose && (
            <button onClick={onClose} className="ai-close-btn" title="Close">
              ✕
            </button>
          )}
        </div>
      </div>

      {error && <Alert type="error" message={error} />}

      <div className="ai-messages-container">
        {messages.length === 0 ? (
          <div className="ai-empty-state">
            <p>Hello! 👋 I'm your Buddy AI assistant.</p>
            <p>
              Ask me questions about tasks, get feedback, or request mentorship.
            </p>
            <div className="ai-quick-hints">
              <button
                onClick={() => setInput("I need help understanding this task")}
                className="hint-button"
              >
                Get Help
              </button>
              <button
                onClick={() => setInput("Can you review my approach?")}
                className="hint-button"
              >
                Review Code
              </button>
              <button
                onClick={() => setInput("What's a good way to start?")}
                className="hint-button"
              >
                Get Started
              </button>
            </div>
          </div>
        ) : (
          <div className="ai-messages">
            {messages.map((msg, idx) => (
              <div key={idx} className={`ai-message ai-message-${msg.sender}`}>
                {msg.sender === "assistant" && (
                  <small className="ai-sender-label">Buddy AI</small>
                )}
                <div className="ai-message-content">
                  <p>{msg.content}</p>
                  {msg.metadata && (
                    <div className="ai-message-metadata">
                      <span className="ai-intent">
                        {msg.metadata.intent?.replace(/_/g, " ")}
                      </span>
                      {msg.metadata.confidence && (
                        <span className="ai-confidence">
                          Confidence:{" "}
                          {(msg.metadata.confidence * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>
                  )}
                </div>
                {msg.timestamp && (
                  <small className="ai-timestamp">
                    {new Date(msg.timestamp).toLocaleTimeString()}
                  </small>
                )}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}

        {loading && (
          <div className="ai-message ai-message-system">
            <Loading size="small" />
          </div>
        )}
      </div>

      <form onSubmit={handleSendMessage} className="ai-input-form">
        <div className="ai-input-wrapper">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask me anything about your tasks..."
            disabled={loading}
            className="ai-input"
            autoComplete="off"
          />
          {hint && <small className="ai-hint">{hint}</small>}
        </div>
        <button
          type="submit"
          disabled={!input.trim() || loading}
          className="btn-send"
        >
          {loading ? "Sending..." : "Send"}
        </button>
      </form>
    </div>
  );
};

export default AIChat;
