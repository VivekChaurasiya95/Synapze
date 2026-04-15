/**
 * AI Service (Frontend)
 * Service for AI API communication
 */

import api from "./api.js";

/**
 * Send message to AI and get response
 * @param {object} params - Request parameters
 * @param {string} params.message - User message
 * @param {string} params.taskId - Optional task ID
 * @param {string} params.submissionId - Optional submission ID
 * @param {array} params.conversationHistory - Conversation history
 * @returns {Promise} - AI response
 */
export const sendMessage = async (params) => {
  const { message, taskId, submissionId, conversationHistory = [] } = params;

  try {
    const response = await api.post("/ai/chat", {
      message,
      taskId,
      submissionId,
      conversationHistory,
    });

    return response.data;
  } catch (error) {
    console.error("Error sending AI message:", {
      status: error?.response?.status,
      data: error?.response?.data,
      message: error?.message,
    });

    // Extract error message from various possible response structures
    const errorData = error?.response?.data;
    if (errorData?.errors && Array.isArray(errorData.errors)) {
      throw { error: errorData.errors.join(", ") };
    } else if (errorData?.error) {
      throw { error: errorData.error };
    } else if (errorData?.message) {
      throw { error: errorData.message };
    } else if (error?.message) {
      throw { error: error.message };
    } else {
      throw { error: "Failed to get AI response. Please try again." };
    }
  }
};

/**
 * Send message to Groq Llama 3.1 via the backend AI route
 * @param {object} params - Request parameters
 * @param {string} params.message - User message
 * @param {string} params.taskId - Optional task ID
 * @param {string} params.submissionId - Optional submission ID
 * @param {array} params.conversationHistory - Conversation history
 * @returns {Promise} - AI response
 */
export const sendGroqMessage = async (params) => {
  const { message, taskId, submissionId, conversationHistory = [] } = params;

  try {
    const response = await api.post("/ai/groq", {
      message,
      taskId,
      submissionId,
      conversationHistory,
    });

    return response.data;
  } catch (error) {
    console.error("Error sending Groq AI message:", {
      status: error?.response?.status,
      data: error?.response?.data,
      message: error?.message,
    });

    const errorData = error?.response?.data;
    if (errorData?.error) {
      throw { error: errorData.error };
    } else if (errorData?.message) {
      throw { error: errorData.message };
    } else if (error?.message) {
      throw { error: error.message };
    } else {
      throw { error: "Failed to get Groq AI response. Please try again." };
    }
  }
};

/**
 * Get conversation history
 * @returns {Promise} - Conversation history
 */
export const getChatHistory = async () => {
  try {
    const response = await api.get("/ai/history");
    return response.data;
  } catch (error) {
    console.error("Error fetching chat history:", error);
    throw error?.response?.data || { error: "Failed to fetch history" };
  }
};

/**
 * Clear conversation history
 * @returns {Promise} - Clear result
 */
export const clearChatHistory = async () => {
  try {
    const response = await api.delete("/ai/history");
    return response.data;
  } catch (error) {
    console.error("Error clearing chat history:", error);
    throw error?.response?.data || { error: "Failed to clear history" };
  }
};

/**
 * Get AI system status
 * @returns {Promise} - System status
 */
export const getAIStatus = async () => {
  try {
    const response = await api.get("/ai/status");
    return response.data;
  } catch (error) {
    console.error("Error fetching AI status:", error);
    throw error?.response?.data || { error: "Failed to fetch status" };
  }
};

/**
 * Check AI health
 * @returns {Promise} - Health status
 */
export const checkAIHealth = async () => {
  try {
    const response = await api.get("/ai/health");
    return response.data;
  } catch (error) {
    console.error("Error checking AI health:", error);
    return { healthy: false };
  }
};

export default {
  sendMessage,
  sendGroqMessage,
  getChatHistory,
  clearChatHistory,
  getAIStatus,
  checkAIHealth,
};
