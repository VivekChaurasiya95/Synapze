/**
 * AI Controller
 * Handles AI-related API requests
 */

import {
  processAIRequest,
  getConversationHistory,
  clearConversationHistory,
  getAISystemStatus,
} from "../services/ai/aiService.js";
import asyncHandler from "../utils/asyncHandler.js";
import logger from "../config/logger.js";

/**
 * POST /api/ai/chat
 * Process chat message and get AI response
 */
export const chatWithAI = asyncHandler(async (req, res) => {
  const { message, taskId, submissionId, conversationHistory } = req.body;
  const { _id: userId, role = "student" } = req.user;

  logger.info("Chat request received", {
    userId,
    role,
    hasMessage: !!message,
  });

  const result = await processAIRequest({
    userId: userId.toString(),
    message,
    role,
    taskId,
    submissionId,
    conversationHistory,
  });

  res.status(result.success ? 200 : result.status || 400).json(result);
});

/**
 * POST /api/ai/groq
 * Process chat message through Groq Llama 3.1
 */
export const chatWithGroqAI = asyncHandler(async (req, res) => {
  const { message, taskId, submissionId, conversationHistory } = req.body;
  const { _id: userId, role = "student" } = req.user;

  logger.info("Groq chat request received", {
    userId,
    role,
    hasMessage: !!message,
  });

  const result = await processAIRequest({
    userId: userId.toString(),
    message,
    role,
    taskId,
    submissionId,
    conversationHistory,
    provider: "groq",
  });

  res.status(result.success ? 200 : 400).json(result);
});

/**
 * GET /api/ai/history
 * Get conversation history for current user
 */
export const getHistory = asyncHandler(async (req, res) => {
  const { _id: userId } = req.user;

  const history = getConversationHistory(userId.toString());

  res.json({
    success: true,
    history,
    count: history.length,
  });
});

/**
 * DELETE /api/ai/history
 * Clear conversation history
 */
export const clearHistory = asyncHandler(async (req, res) => {
  const { _id: userId } = req.user;

  clearConversationHistory(userId.toString());

  logger.info("Conversation history cleared", { userId });

  res.json({
    success: true,
    message: "Conversation history cleared",
  });
});

/**
 * GET /api/ai/status
 * Get AI system status
 */
export const getStatus = asyncHandler(async (req, res) => {
  const status = await getAISystemStatus();

  res.json({
    success: true,
    status,
  });
});

/**
 * GET /api/ai/health
 * Health check endpoint
 */
export const healthCheck = asyncHandler(async (req, res) => {
  const status = await getAISystemStatus();

  res.status(status.healthy ? 200 : 503).json({
    healthy: status.healthy,
    status: status.ollama?.status || "unknown",
  });
});

export default {
  chatWithAI,
  getHistory,
  clearHistory,
  getStatus,
  healthCheck,
};
