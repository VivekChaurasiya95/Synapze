/**
 * AI Context
 * React Context for AI chat state management
 */

import { createContext, useContext, useState, useCallback } from "react";
import { sendMessage, getChatHistory } from "../services/aiService.js";

const AIContext = createContext(null);

export const AIProvider = ({ children }) => {
  const [messages, setMessages] = useState([]);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentSession, setCurrentSession] = useState(null);

  /**
   * Initialize AI context
   */
  const initializeAI = useCallback(async () => {
    try {
      setLoading(true);
      const result = await getChatHistory();
      if (result.success) {
        setHistory(result.history || []);
        setMessages(result.history || []);
        setCurrentSession({
          startedAt: new Date(),
          messageCount: result.history?.length || 0,
        });
      }
    } catch (err) {
      setError(err?.error || "Failed to initialize AI");
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * Send message to AI
   */
  const handleSendMessage = useCallback(
    async (message, taskId, submissionId) => {
      const trimmedMessage = message?.trim();
      if (!trimmedMessage || trimmedMessage.length < 3) {
        throw new Error("Please type at least 3 characters before sending.");
      }

      try {
        setError(null);
        setLoading(true);

        // Add user message
        const userMsg = {
          sender: "user",
          content: trimmedMessage,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, userMsg]);

        // Get AI response
        const result = await sendMessage({
          message: trimmedMessage,
          taskId,
          submissionId,
          conversationHistory: messages,
        });

        if (result.success) {
          const assistantMsg = {
            sender: "assistant",
            content: result.response,
            metadata: result.metadata,
            timestamp: new Date(),
          };
          setMessages((prev) => [...prev, assistantMsg]);
          return result;
        } else {
          setError(result.error);
          const errorMsg = {
            sender: "system",
            content: `Error: ${result.error}`,
            timestamp: new Date(),
          };
          setMessages((prev) => [...prev, errorMsg]);
          throw new Error(result.error);
        }
      } catch (err) {
        const errorMsg = err.message || "Failed to send message";
        setError(errorMsg);
        throw err;
      } finally {
        setLoading(false);
      }
    },
    [messages],
  );

  /**
   * Clear messages
   */
  const clearMessages = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  /**
   * Get last message
   */
  const getLastMessage = useCallback(() => {
    return messages[messages.length - 1] || null;
  }, [messages]);

  /**
   * Get user messages count
   */
  const getUserMessageCount = useCallback(() => {
    return messages.filter((msg) => msg.sender === "user").length;
  }, [messages]);

  /**
   * Get ai responses count
   */
  const getAIResponseCount = useCallback(() => {
    return messages.filter((msg) => msg.sender === "assistant").length;
  }, [messages]);

  const value = {
    // State
    messages,
    history,
    loading,
    error,
    currentSession,

    // Methods
    initializeAI,
    handleSendMessage,
    clearMessages,
    getLastMessage,
    getUserMessageCount,
    getAIResponseCount,
    setError,
  };

  return <AIContext.Provider value={value}>{children}</AIContext.Provider>;
};

/**
 * Hook to use AI context
 */
export const useAI = () => {
  const context = useContext(AIContext);
  if (!context) {
    throw new Error("useAI must be used within AIProvider");
  }
  return context;
};

export default AIContext;
