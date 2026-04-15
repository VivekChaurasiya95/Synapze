/**
 * AI Routes
 * REST endpoints for AI functionality
 */

import { Router } from "express";
import { protect } from "../middleware/auth.js";
import { chatAIRules } from "../middleware/validate.js";
import {
  chatWithAI,
  chatWithGroqAI,
  getHistory,
  clearHistory,
  getStatus,
  healthCheck,
} from "../controllers/aiController.js";

const router = Router();

/**
 * Public endpoints
 */
router.get("/health", healthCheck);

/**
 * Protected endpoints (require authentication)
 */
router.use(protect);

// Main chat endpoint
router.post("/chat", chatAIRules, chatWithAI);

// Groq-specific chat endpoint
router.post("/groq", chatAIRules, chatWithGroqAI);

// Get conversation history
router.get("/history", getHistory);

// Clear conversation history
router.delete("/history", clearHistory);

// Get AI system status
router.get("/status", getStatus);

export default router;
