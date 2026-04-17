import "dotenv/config";
import express from "express";
import cors from "cors";
import helmet from "helmet";
import cookieParser from "cookie-parser";
import { createServer } from "http";
import { Server } from "socket.io";
import jwt from "jsonwebtoken";
import mongoose from "mongoose";
import connectDB from "./config/db.js";
import errorHandler from "./middleware/errorHandler.js";
import { apiLimiter } from "./middleware/rateLimiter.js";
import { sanitizeMiddleware } from "./utils/sanitize.js";
import requestTimeout from "./middleware/requestTimeout.js";
import socketAuthAndRateLimit from "./middleware/socketRateLimit.js";
import logger from "./config/logger.js";
import { ensureDummyAdmin } from "./utils/adminBootstrap.js";

// Route imports
import authRoutes from "./routes/authRoutes.js";
import taskRoutes from "./routes/taskRoutes.js";
import userRoutes from "./routes/userRoutes.js";
import chatRoutes from "./routes/chatRoutes.js";
import mentorRoutes from "./routes/mentorRoutes.js";
import metricsRoutes from "./routes/metricsRoutes.js";
import aiRoutes from "./routes/aiRoutes.js";
import adminRoutes from "./admin/admin.routes.js";
import { metricsMiddleware } from "./middleware/metrics.js";

// Load env vars - handled by import "dotenv/config"
// dotenv.config();

const app = express();
const httpServer = createServer(app);

// Trust the first proxy hop (Render, Railway, Heroku, Nginx, etc.)
// Required for express-rate-limit to see the real client IP via X-Forwarded-For
if (process.env.NODE_ENV === "production") {
  app.set("trust proxy", 1);
}

// Disable etag generation to prevent 304 responses on GET requests
app.set("etag", false);

// Health check endpoint at root (for Render health checks - BEFORE CORS middleware)
app.get("/", (req, res) => {
  res.status(200).json({ status: "ok", message: "Server is running" });
});

app.head("/", (req, res) => {
  res.status(200).end();
});

// Allowed origins for CORS
const allowedOrigins =
  process.env.NODE_ENV === "production"
    ? [process.env.CLIENT_URL]
    : [
        process.env.CLIENT_URL,
        "http://localhost:5173",
        "http://localhost:5174",
      ].filter(Boolean);

// Socket.IO setup with connection limits and timeouts
const io = new Server(httpServer, {
  cors: {
    origin: allowedOrigins,
    methods: ["GET", "POST"],
    credentials: true,
    allowEIO3: true,  // Allow Engine.IO v3 clients for compatibility
  },
  maxHttpBufferSize: 1e6, // 1MB max for socket messages
  pingInterval: 25000, // Ping clients every 25s
  pingTimeout: 60000, // Timeout after 60s no pong
  transports: ['websocket', 'polling'],  // Support both websocket and polling
});

// --- Security middleware ---
app.use(helmet());

// Cookie parser (required to read httpOnly auth cookies)
app.use(cookieParser());

// Body parser
app.use(express.json({ limit: "10kb" })); // Limit body size

// Enable CORS
app.use(
  cors({
    origin: (origin, callback) => {
      // Allow requests with no origin (e.g. curl, Postman) in dev only
      if (!origin && process.env.NODE_ENV !== "production")
        return callback(null, true);
      if (allowedOrigins.includes(origin)) return callback(null, true);
      callback(new Error("Not allowed by CORS"));
    },
    credentials: true,
  }),
);

// Sanitirequest timeout middleware (30 seconds default)
app.use(requestTimeout(30000));

// Metrics collection middleware (tracks requests, memory, connection counts)
app.use(metricsMiddleware);

// Apply general rate limiting to all API routes
app.use("/api", apiLimiter);

// Request logging in development and production
app.use((req, res, next) => {
  const start = Date.now();
  res.on("finish", () => {
    const duration = Date.now() - start;
    logger.debug(
      `${req.method} ${req.path} - ${res.statusCode} (${duration}ms)`,
    );
  });
  next();
});

// Apply general rate limiting to all API routes
app.use("/api", apiLimiter);

// Mount routers
app.use("/api/auth", authRoutes);
app.use("/api/tasks", taskRoutes);
app.use("/api/users", userRoutes);
app.use("/api/chat", chatRoutes);
app.use("/api/mentors", mentorRoutes);
app.use("/api/metrics", metricsRoutes);
app.use("/api/ai", aiRoutes);
app.use("/api/admin", adminRoutes);

// Health check route (verifies DB connectivity)
app.get("/api/health", async (req, res) => {
  try {
    const dbState = mongoose.connection.readyState;
    const dbStatus = [
      "disconnected",
      "connected",
      "connecting",
      "disconnecting",
    ];
    res.status(dbState === 1 ? 200 : 503).json({
      success: dbState === 1,
      message: "Server is running",
      database: dbStatus[dbState] || "unknown",
      uptime: process.uptime(),
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    res.status(503).json({
      success: false,
      message: "Server error",
      error: error.message,
    });
  }
});

// Error handling middleware
app.use(errorHandler);

// --- Socket.IO middleware ---
// Rate limiting and max connections
io.use(socketAuthAndRateLimit(io));

// JWT authentication
io.use((socket, next) => {
  const token = socket.handshake.auth?.token || socket.handshake.query?.token;
  if (!token) {
    return next(new Error("Authentication error: No token provided"));
  }
  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    socket.userId = decoded.id;
    socket.connectedAt = Date.now();
    next();
  } catch (err) {
    logger.error("Socket authentication failed", err, { socketId: socket.id });
    return next(new Error("Authentication error: Invalid token"));
  }
});

// Socket.IO connection handling
const connectedUsers = new Map();

io.on("connection", (socket) => {
  console.log("User connected:", socket.id, "| userId:", socket.userId);

  // Auto-join user to their personal room using authenticated userId
  connectedUsers.set(socket.userId, socket.id);
  socket.join(socket.userId);

  // Join chat room
  socket.on("chat:join", (chatRoomId) => {
    if (!chatRoomId || typeof chatRoomId !== "string") return;
    socket.join(chatRoomId);
  });

  // Leave chat room
  socket.on("chat:leave", (chatRoomId) => {
    if (!chatRoomId || typeof chatRoomId !== "string") return;
    socket.leave(chatRoomId);
  });

  // Send message
  socket.on("chat:message", (data) => {
    if (!data || !data.chatRoomId || !data.message) return;
    if (typeof data.message !== "string") return;
    // Enforce maximum message length to prevent large payload abuse
    const MAX_MSG_LEN = 2000;
    const { chatRoomId } = data;
    const message =
      typeof data.message === "string"
        ? data.message.trim().slice(0, MAX_MSG_LEN)
        : "";
    if (!message) return;
    socket.to(chatRoomId).emit("chat:newMessage", message);
  });

  // Typing indicator
  socket.on("chat:typing", (data) => {
    if (!data || !data.chatRoomId) return;
    const { chatRoomId, user } = data;
    socket.to(chatRoomId).emit("chat:userTyping", user);
  });

  // Stop typing indicator
  socket.on("chat:stopTyping", (data) => {
    if (!data || !data.chatRoomId) return;
    const { chatRoomId, user } = data;
    socket.to(chatRoomId).emit("chat:userStopTyping", user);
  });

  // Read receipt
  socket.on("chat:read", (data) => {
    if (!data || !data.chatRoomId) return;
    const { chatRoomId } = data;
    socket.to(chatRoomId).emit("chat:messagesRead", { userId: socket.userId });
  });

  // Task notifications — use authenticated userId as sender
  socket.on("task:taken", (data) => {
    if (!data || !data.taskPosterId || !data.message) return;
    const { taskPosterId, message } = data;
    io.to(taskPosterId).emit("notification", {
      type: "task_taken",
      message,
    });
  });

  socket.on("task:submitted", (data) => {
    if (!data || !data.taskPosterId || !data.message) return;
    const { taskPosterId, message } = data;
    io.to(taskPosterId).emit("notification", {
      type: "task_submitted",
      message,
    });
  });

  socket.on("task:completed", (data) => {
    if (!data || !data.userId || !data.message) return;
    const { userId, message, credits } = data;
    io.to(userId).emit("notification", {
      type: "task_completed",
      message,
      credits,
    });
  });

  // Handle socket errors
  socket.on("error", (error) => {
    logger.error("Socket error", error, {
      socketId: socket.id,
      userId: socket.userId,
    });
  });

  // Disconnect
  socket.on("disconnect", () => {
    connectedUsers.delete(socket.userId);
    const connectedSeconds = (Date.now() - socket.connectedAt) / 1000;
    logger.debug(
      `User disconnected: ${socket.id} (connected for ${connectedSeconds}s)`,
    );
  });
});

// Make io accessible to routes
app.set("io", io);

// --- Graceful shutdown ---
const gracefulShutdown = (signal) => {
  logger.warn(`${signal} received. Shutting down gracefully...`);

  // Close HTTP server
  httpServer.close(() => {
    logger.info("HTTP server closed");
    mongoose.connection.close(false).then(() => {
      logger.info("MongoDB connection closed");
      process.exit(0);
    });
  });

  // Close all active socket connections
  io.close();

  // Force shutdown after 30 seconds
  setTimeout(() => {
    logger.critical("Forced shutdown after timeout");
    process.exit(1);
  }, 30000);
};

process.on("SIGTERM", () => gracefulShutdown("SIGTERM"));
process.on("SIGINT", () => gracefulShutdown("SIGINT"));

// Handle uncaught exceptions
process.on("uncaughtException", (error) => {
  logger.critical("Uncaught exception detected", error);
  gracefulShutdown("UNCAUGHT_EXCEPTION");
});

// Handle unhandled promise rejections
process.on("unhandledRejection", (reason, promise) => {
  logger.error("Unhandled promise rejection", new Error(String(reason)), {
    promise,
  });
});

// Start server
const startServer = async () => {
  try {
    await connectDB();
    logger.info("Database connected successfully");

    await ensureDummyAdmin();

    const PORT = process.env.PORT || 5000;
    httpServer.listen(PORT, () => {
      logger.info(
        `Server running on port ${PORT} in ${process.env.NODE_ENV || "development"} mode`,
      );
    });

    httpServer.on("error", (error) => {
      logger.critical("HTTP server error", error);
    });
  } catch (error) {
    logger.critical("Failed to start server", error);
    process.exit(1);
  }
};

// Start the server
startServer();

// Export for testing
export { app, httpServer, io, startServer };
