/**
 * AI Configuration Module
 * Manages Groq AI connection and model settings
 */

import Groq from 'groq-sdk';
import logger from '../config/logger.js';

// Groq Configuration
const GROQ_CONFIG = {
  apiKey: process.env.GROQ_API_KEY?.trim(),
  model: process.env.GROQ_MODEL || 'llama-3.1-8b-instant',
  timeout: parseInt(process.env.GROQ_TIMEOUT || '30000', 10),
  maxTokens: parseInt(process.env.GROQ_MAX_TOKENS || '1024', 10),
  temperature: parseFloat(process.env.GROQ_TEMPERATURE || '0.7'),
};

// Initialize Groq client
const groqClient = new Groq({
  apiKey: GROQ_CONFIG.apiKey,
});

/**
 * Check if Groq service is available
 * @returns {Promise<boolean>} - Connection status
 */
export const checkAIConnection = async () => {
  if (!GROQ_CONFIG.apiKey) {
    logger.error('Groq API key not configured');
    return false;
  }

  try {
    // Simple health check by attempting a minimal request
    await groqClient.chat.completions.create({
      messages: [{ role: 'user', content: 'Hello' }],
      model: GROQ_CONFIG.model,
      max_tokens: 1,
    });
    logger.info('Groq service is healthy');
    return true;
  } catch (error) {
    logger.error('Groq service health check failed', {
      error: error.message,
      model: GROQ_CONFIG.model,
    });

    // Handle specific error types
    if (error.status === 401) {
      logger.error('Invalid Groq API key');
    } else if (error.status === 429) {
      logger.error('Groq rate limit exceeded');
    } else if (error.code === 'ECONNREFUSED' || error.message.includes('fetch')) {
      logger.error('Cannot connect to Groq API');
    }

    return false;
  }
};

/**
 * Generate AI response using Groq
 * @param {string} prompt - The prompt to send to the model
 * @param {object} options - Generation options
 * @returns {Promise<string>} - AI model response
 */
export const generateAIResponse = async (prompt, options = {}) => {
  if (!GROQ_CONFIG.apiKey) {
    throw new Error('Missing Groq API key. Set GROQ_API_KEY in environment variables.');
  }

  try {
    logger.info('Sending request to Groq', {
      model: GROQ_CONFIG.model,
      promptLength: prompt.length,
    });

    const response = await groqClient.chat.completions.create({
      messages: [{ role: 'user', content: prompt }],
      model: GROQ_CONFIG.model,
      temperature: options.temperature ?? GROQ_CONFIG.temperature,
      max_tokens: options.numPredict ?? GROQ_CONFIG.maxTokens,
    });

    const content = response.choices[0]?.message?.content;

    if (!content) {
      const errorMsg = 'Invalid response from Groq: No content received';
      logger.error(errorMsg, { response });
      throw new Error(errorMsg);
    }

    logger.info('Received response from Groq', {
      responseLength: content.length,
    });

    return content.trim();
  } catch (error) {
    logger.error('Error generating AI response', {
      error: error.message,
      model: GROQ_CONFIG.model,
      status: error.status,
    });

    // Handle specific error types
    if (error.status === 401) {
      throw new Error('Invalid Groq API key. Please check your GROQ_API_KEY environment variable.');
    }

    if (error.status === 429) {
      throw new Error('Groq rate limit exceeded. Please try again later.');
    }

    if (error.status === 404) {
      throw new Error(`Groq model '${GROQ_CONFIG.model}' not found. Please verify GROQ_MODEL environment variable.`);
    }

    if (error.code === 'ECONNREFUSED' || error.message.includes('fetch') || error.message.includes('network')) {
      throw new Error('Cannot connect to Groq API. Please check your internet connection and GROQ_API_KEY.');
    }

    // Generic error
    throw new Error(`Groq API error: ${error.message}`);
  }
};

export default {
  GROQ_CONFIG,
  groqClient,
  checkAIConnection,
  generateAIResponse,
};
