/**
 * AI Configuration Module
 * Manages Ollama connection and AI model settings
 */

import axios from "axios";
import logger from "../config/logger.js";

// Ollama Configuration
const OLLAMA_CONFIG = {
  host: process.env.OLLAMA_HOST || "http://localhost:11434",
  model: process.env.AI_MODEL || "mistral",
  timeout: parseInt(process.env.AI_TIMEOUT || "30000", 10),
  maxTokens: parseInt(process.env.AI_MAX_TOKENS || "1024", 10),
  temperature: parseFloat(process.env.AI_TEMPERATURE || "0.7"),
};

const GROQ_CONFIG = {
  baseURL: process.env.GROQ_BASE_URL || "https://api.groq.com/v1",
  model: process.env.GROQ_MODEL || "llama_3_1",
  timeout: parseInt(process.env.GROQ_TIMEOUT || "30000", 10),
  maxTokens: parseInt(process.env.GROQ_MAX_TOKENS || "1024", 10),
  temperature: parseFloat(process.env.GROQ_TEMPERATURE || "0.7"),
  apiKey: process.env.GROQ_API_KEY?.trim() || "",
};

const AI_PROVIDER = process.env.AI_PROVIDER
  ? process.env.AI_PROVIDER.trim().toLowerCase()
  : GROQ_CONFIG.apiKey
  ? "groq"
  : "ollama";

// Create Ollama API client
const ollamaClient = axios.create({
  baseURL: OLLAMA_CONFIG.host,
  timeout: OLLAMA_CONFIG.timeout,
  headers: {
    "Content-Type": "application/json",
  },
});

const groqHeaders = {
  "Content-Type": "application/json",
};

if (GROQ_CONFIG.apiKey) {
  groqHeaders.Authorization = `Bearer ${GROQ_CONFIG.apiKey}`;
}

const groqClient = axios.create({
  baseURL: GROQ_CONFIG.baseURL,
  timeout: GROQ_CONFIG.timeout,
  headers: groqHeaders,
});

/**
 * Check if Ollama service is available
 */
export const checkOllamaHealth = async () => {
  try {
    const response = await ollamaClient.get("/api/tags");
    logger.info("Ollama service is healthy", {
      models: response.data.models?.length || 0,
    });
    return true;
  } catch (error) {
    logger.error("Ollama service health check failed", {
      error: error.message,
      host: OLLAMA_CONFIG.host,
    });
    return false;
  }
};

export const checkGroqHealth = async () => {
  if (!GROQ_CONFIG.apiKey) {
    return false;
  }

  try {
    const response = await sendGroqRequest({
      input: 'Say "OK" to confirm connection.',
      temperature: 0.1,
      max_output_tokens: 1,
    });
    return Boolean(response?.data);
  } catch (error) {
    logger.error("Groq service health check failed", {
      error: error.message,
      baseURL: GROQ_CONFIG.baseURL,
      model: GROQ_CONFIG.model,
    });
    return false;
  }
};

const extractGroqText = (payload) => {
  if (!payload) {
    return null;
  }

  if (typeof payload === "string") {
    return payload;
  }

  if (Array.isArray(payload)) {
    return payload
      .map((item) => extractGroqText(item))
      .filter(Boolean)
      .join(" ");
  }

  if (typeof payload === "object") {
    if (typeof payload.text === "string") {
      return payload.text;
    }
    if (typeof payload.output === "string") {
      return payload.output;
    }
    if (Array.isArray(payload.output)) {
      return extractGroqText(payload.output);
    }
    if (Array.isArray(payload.content)) {
      return extractGroqText(payload.content);
    }
    if (payload.content) {
      return extractGroqText(payload.content);
    }
  }

  return null;
};

const sendGroqRequest = async (payload) => {
  try {
    return await groqClient.post(`/models/${GROQ_CONFIG.model}`, payload);
  } catch (error) {
    if (error.response?.status === 404) {
      return await groqClient.post(`/models/${GROQ_CONFIG.model}/infer`, payload);
    }
    throw error;
  }
};

export const checkAIConnection = async () => {
  if (AI_PROVIDER === "groq") {
    return checkGroqHealth();
  }

  return checkOllamaHealth();
};

/**
 * Send request to Ollama AI model
 * @param {string} prompt - The prompt to send to the model
 * @param {object} options - Generation options
 * @returns {Promise<string>} - AI model response
 */
export const generateGroqResponse = async (prompt, options = {}) => {
  if (!GROQ_CONFIG.apiKey) {
    throw new Error("Missing Groq API key. Set GROQ_API_KEY in environment.");
  }

  const requestPayload = {
    input: prompt,
    temperature: options.temperature ?? GROQ_CONFIG.temperature,
    max_output_tokens: options.numPredict || GROQ_CONFIG.maxTokens,
  };

  logger.info("Sending request to Groq", {
    model: GROQ_CONFIG.model,
    provider: "groq",
    promptLength: prompt.length,
    baseURL: GROQ_CONFIG.baseURL,
  });

  const response = await sendGroqRequest(requestPayload);
  const rawOutput = extractGroqText(response.data);

  if (!rawOutput) {
    const errorMsg = `Invalid response from Groq: ${JSON.stringify(response.data)}`;
    logger.error(errorMsg);
    throw new Error(errorMsg);
  }

  logger.info("Received response from Groq", {
    responseLength: rawOutput.length,
  });

  return rawOutput.trim();
};

export const generateAIResponse = async (prompt, options = {}) => {
  const provider = options.provider?.toLowerCase() || AI_PROVIDER;

  try {
    if (provider === "groq") {
      return await generateGroqResponse(prompt, options);
    }

    const requestPayload = {
      model: OLLAMA_CONFIG.model,
      prompt: prompt,
      stream: false,
      temperature: options.temperature || OLLAMA_CONFIG.temperature,
      top_p: options.topP || 0.9,
      top_k: options.topK || 40,
      num_predict: options.numPredict || OLLAMA_CONFIG.maxTokens,
    };

    logger.info("Sending request to Ollama", {
      model: OLLAMA_CONFIG.model,
      promptLength: prompt.length,
      host: OLLAMA_CONFIG.host,
    });

    const response = await ollamaClient.post("/api/generate", requestPayload);

    if (!response.data || !response.data.response) {
      const errorMsg = `Invalid response from Ollama: ${JSON.stringify(response.data)}`;
      logger.error(errorMsg);
      throw new Error(errorMsg);
    }

    logger.info("Received response from Ollama", {
      responseLength: response.data.response.length,
    });

    return response.data.response.trim();
  } catch (error) {
    const host = provider === "groq" ? GROQ_CONFIG.baseURL : OLLAMA_CONFIG.host;
    logger.error("Error generating AI response", {
      provider,
      error: error.message,
      model: provider === "groq" ? GROQ_CONFIG.model : OLLAMA_CONFIG.model,
      host,
      status: error.response?.status,
      statusText: error.response?.statusText,
      errorData: error.response?.data,
    });

    // Provide more context about the error
    if (
      error.code === "ECONNREFUSED" ||
      error.message.includes("ECONNREFUSED")
    ) {
      throw new Error(
        provider === "groq"
          ? `Cannot connect to Groq at ${host}. Please verify GROQ_BASE_URL and network access.`
          : `Cannot connect to Ollama at ${host}. Please start Ollama service.`,
      );
    }

    if (provider === "ollama" && (error.response?.status === 404 || error.message.includes("model"))) {
      throw new Error(
        `Ollama model '${OLLAMA_CONFIG.model}' not found. Please pull it using: ollama pull ${OLLAMA_CONFIG.model}`,
      );
    }

    if (provider === "groq" && error.response?.status === 404) {
      throw new Error(
        `Groq model '${GROQ_CONFIG.model}' not found. Verify GROQ_MODEL and GROQ_BASE_URL settings.`,
      );
    }

    if (error.code === "ENOTFOUND") {
      throw new Error(
        provider === "groq"
          ? `Cannot reach Groq host: ${host}. Check your GROQ_BASE_URL environment variable.`
          : `Cannot reach Ollama host: ${host}. Check your OLLAMA_HOST environment variable.`,
      );
    }

    throw new Error(
      provider === "groq"
        ? `Groq Error: ${error.message}`
        : `Ollama Error: ${error.message}`,
    );
  }
};

/**
 * Stream AI response (for real-time output)
 * @param {string} prompt - The prompt to send
 * @param {function} onData - Callback for each token
 * @param {object} options - Generation options
 */
export const streamAIResponse = async (prompt, onData, options = {}) => {
  try {
    const requestPayload = {
      model: OLLAMA_CONFIG.model,
      prompt: prompt,
      stream: true,
      temperature: options.temperature || OLLAMA_CONFIG.temperature,
    };

    const response = await ollamaClient.post("/api/generate", requestPayload, {
      responseType: "stream",
    });

    response.data.on("data", (chunk) => {
      try {
        const lines = chunk.toString().split("\n");
        lines.forEach((line) => {
          if (line.trim()) {
            const data = JSON.parse(line);
            if (data.response) {
              onData(data.response);
            }
          }
        });
      } catch (error) {
        logger.error("Error parsing stream data", { error: error.message });
      }
    });

    response.data.on("error", (error) => {
      logger.error("Stream error", { error: error.message });
    });

    return new Promise((resolve, reject) => {
      response.data.on("end", () => resolve());
      response.data.on("error", reject);
    });
  } catch (error) {
    logger.error("Error streaming AI response", { error: error.message });
    throw error;
  }
};

/**
 * Get Ollama model list
 * @returns {Promise<array>} - List of available models
 */
export const getAvailableModels = async () => {
  try {
    const response = await ollamaClient.get("/api/tags");
    return response.data.models || [];
  } catch (error) {
    logger.error("Error fetching available models", { error: error.message });
    return [];
  }
};

/**
 * Test AI connection
 * @returns {Promise<boolean>} - Connection status
 */
export const testAIConnection = async () => {
  try {
    const prompt = 'Say "OK" to confirm connection.';
    const response = await generateAIResponse(prompt, {
      temperature: 0.1,
      numPredict: 50,
    });
    return response.toLowerCase().includes("ok");
  } catch (error) {
    logger.error("AI connection test failed", { error: error.message });
    return false;
  }
};

export default {
  ollamaClient,
  OLLAMA_CONFIG,
  GROQ_CONFIG,
  AI_PROVIDER,
  checkOllamaHealth,
  checkGroqHealth,
  checkAIConnection,
  generateAIResponse,
  streamAIResponse,
  getAvailableModels,
  testAIConnection,
};
