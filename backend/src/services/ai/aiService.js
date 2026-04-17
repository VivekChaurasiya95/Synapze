/**
 * AI Orchestration Service
 * Main orchestrator that coordinates all AI components
 */

import {
  generateAIResponse,
  checkAIConnection,
} from "../../utils/aiConfig.js";
import { detectIntent } from "./intentDetection.js";
import {
  buildContext,
  formatContextForPrompt,
  sanitizeContext,
} from "./contextBuilder.js";
import {
  moderateMessage,
  moderateResponse,
  validateRateLimit,
  createSafeResponse,
} from "./moderationService.js";
import {
  constructPrompt,
  createSystemMessage,
  buildConversationHistory,
} from "./promptConstructor.js";
import {
  canAccessIntent,
  adaptResponseForRole,
  getRoleContext,
} from "./roleBehavior.js";
import logger from "../../config/logger.js";

// Store for conversation history and rate limiting
const conversationStore = new Map();
const rateLimitStore = new Map();

/**
 * Process AI request through full orchestration pipeline
 * @param {object} params - Request parameters
 * @returns {Promise<object>} - AI response
 */
export const processAIRequest = async (params = {}) => {
  try {
    const {
      userId,
      message,
      role = "student",
      taskId,
      submissionId,
      sessionId,
      conversationHistory = [],
    } = params;

    const requestId = `${userId}-${Date.now()}`;

    logger.info("Processing AI request", {
      userId,
      role,
      requestId,
      messageLength: message?.length,
    });

    // Step 1: Validate input
    if (!message || !userId) {
      throw new Error("Missing required parameters: message and userId");
    }

    // Step 2: Moderate input message
    const messageModeration = moderateMessage(message);
    if (!messageModeration.approved) {
      return {
        success: false,
        error: messageModeration.reason,
        severity: messageModeration.severity,
      };
    }

    // Step 3: Check rate limits
    const rateCheckResult = checkRateLimit(userId, role);
    if (!rateCheckResult.allowed) {
      return {
        success: false,
        error: rateCheckResult.reason,
        retryAfter: rateCheckResult.retryAfter,
      };
    }

    // Step 4: Detect intent
    const intentResult = detectIntent(message, {
      role,
      taskId,
    });

    // Step 5: Check role permissions
    if (!canAccessIntent(role, intentResult.intent)) {
      return {
        success: false,
        error: `Your role (${role}) cannot access ${intentResult.intent} assistance`,
      };
    }

    // Step 6: Build context
    const context = await buildContext({
      userId,
      taskId,
      submissionId,
      role,
    });

    // Step 7: Construct prompt
    const prompt = constructPrompt({
      message,
      intent: intentResult.intent,
      context: sanitizeContext(context),
      role,
    });

    // Add conversation history
    const conversationHistoryStr = buildConversationHistory(
      conversationHistory,
      5,
    );

    // Step 8: Generate AI response
    const aiResponse = await generateAIResponse(
      prompt + conversationHistoryStr,
      {
        temperature: 0.7,
        numPredict: 1024,
      },
    );

    // Step 9: Moderate response
    const responseModeration = moderateResponse(aiResponse);
    if (!responseModeration.approved) {
      logger.warn("Response failed moderation", {
        requestId,
        reason: responseModeration.reason,
      });
      return {
        success: false,
        error: "Response generation failed validation",
        reason: responseModeration.reason,
      };
    }

    // Step 10: Adapt for role
    const adaptedResponse = adaptResponseForRole(
      responseModeration.cleanedResponse,
      role,
      { intent: intentResult.intent },
    );

    // Step 11: Add safety wrapper
    const finalResponse = createSafeResponse(adaptedResponse, {
      intent: intentResult.intent,
      role,
    });

    // Step 12: Store in conversation history
    storeConversation(userId, {
      sender: "user",
      content: message,
      intent: intentResult.intent,
      timestamp: new Date(),
    });

    storeConversation(userId, {
      sender: "assistant",
      content: finalResponse,
      timestamp: new Date(),
    });

    // Update rate limit
    incrementRateLimit(userId);

    logger.info("AI request processed successfully", {
      userId,
      role,
      intent: intentResult.intent,
      requestId,
    });

    return {
      success: true,
      response: finalResponse,
      metadata: {
        intent: intentResult.intent,
        confidence: intentResult.confidence,
        role: role,
        warnings: responseModeration.warnings || [],
      },
    };
  } catch (error) {
    logger.error("Error processing AI request", {
      error: error.message,
      userId: params.userId,
      stack: error.stack,
    });

    // Provide more specific error messages
    let errorMessage =
      "An error occurred processing your request. Please try again.";

    if (error.message.includes("Ollama")) {
      errorMessage = "AI service is not available. Please try again later.";
    } else if (
      error.message.includes("timeout") ||
      error.message.includes("ECONNREFUSED")
    ) {
      errorMessage =
        "Connection to AI service failed. Please check if the AI service is running.";
    } else if (error.message.includes("model")) {
      errorMessage = "AI model is not available. Please contact administrator.";
    }

    let statusCode = 500;

    if (error.message.includes("Missing required parameters")) {
      statusCode = 400;
    } else if (
      error.message.includes("Ollama") ||
      error.message.includes("Groq") ||
      error.message.includes("Connection to AI service failed") ||
      error.message.includes("Cannot connect") ||
      error.message.includes("AI service is not available")
    ) {
      statusCode = 503;
    }

    return {
      success: false,
      error: errorMessage,
      reason: error.message,
      status: statusCode,
    };
  }
};

/**
 * Check rate limit for user
 * @param {string} userId - User ID
 * @param {string} role - User role
 * @returns {object} - Rate limit check result
 */
const checkRateLimit = (userId, role) => {
  const rateKey = `${userId}-${role}`;
  const config = getRoleContext(role);
  const maxRequests = config.rateLimit || 50;

  if (!rateLimitStore.has(rateKey)) {
    rateLimitStore.set(rateKey, {
      requests: [],
      firstRequest: Date.now(),
    });
  }

  const record = rateLimitStore.get(rateKey);
  const oneHourAgo = Date.now() - 60 * 60 * 1000;

  // Clean old requests
  record.requests = record.requests.filter((time) => time > oneHourAgo);

  if (record.requests.length >= maxRequests) {
    const oldestRequest = Math.min(...record.requests);
    const retryAfter = Math.ceil(
      (oldestRequest + 60 * 60 * 1000 - Date.now()) / 1000,
    );

    return {
      allowed: false,
      reason: `Rate limit exceeded. Max ${maxRequests} requests per hour.`,
      retryAfter,
    };
  }

  return { allowed: true };
};

/**
 * Increment rate limit counter
 * @param {string} userId - User ID
 */
const incrementRateLimit = (userId) => {
  for (const [key, record] of rateLimitStore.entries()) {
    if (key.startsWith(userId)) {
      record.requests.push(Date.now());
      break;
    }
  }
};

/**
 * Store conversation message
 * @param {string} userId - User ID
 * @param {object} message - Message to store
 */
const storeConversation = (userId, message) => {
  if (!conversationStore.has(userId)) {
    conversationStore.set(userId, []);
  }

  const conversation = conversationStore.get(userId);
  conversation.push(message);

  // Keep only last 50 messages
  if (conversation.length > 50) {
    conversation.shift();
  }
};

/**
 * Get conversation history for user
 * @param {string} userId - User ID
 * @returns {array} - Conversation messages
 */
export const getConversationHistory = (userId) => {
  return conversationStore.get(userId) || [];
};

/**
 * Clear conversation history
 * @param {string} userId - User ID
 */
export const clearConversationHistory = (userId) => {
  conversationStore.delete(userId);
};

/**
 * Get AI system status
 * @returns {Promise<object>} - System status
 */
export const getAISystemStatus = async () => {
  try {
    const isHealthy = await checkAIConnection();
    const provider = "groq";

    return {
      healthy: isHealthy,
      ai: {
        provider,
        status: isHealthy ? "running" : "offline",
        model: process.env.GROQ_MODEL || "llama-3.1-8b-instant",
      },
      activeConversations: conversationStore.size,
      timestamp: new Date(),
    };
  } catch (error) {
    logger.error("Error checking AI system status", { error: error.message });
    return {
      healthy: false,
      error: error.message,
      timestamp: new Date(),
    };
  }
};

/**
 * Import role behavior utilities
 */

export default {
  processAIRequest,
  getConversationHistory,
  clearConversationHistory,
  getAISystemStatus,
};
