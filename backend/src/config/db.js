import mongoose from "mongoose";
import dns from "node:dns";
import { MongoMemoryServer } from 'mongodb-memory-server';
import logger from "./logger.js";

// Connection pool configuration for better resource management
const MAX_RETRIES = 3; // Reduced for faster feedback
const RETRY_DELAY = 5000; // 5 seconds
const CONNECTION_POOL_SIZE = 10; // Max connections in pool
const CONNECTION_TIMEOUT = 30000; // 30 seconds to establish connection

let mongodInstance = null;

const configureDnsForSrv = () => {
  const mongoUri = process.env.MONGODB_URI || "";
  if (!mongoUri.startsWith("mongodb+srv://")) return;

  const dnsServers = (process.env.DNS_SERVERS || "8.8.8.8,1.1.1.1")
    .split(",")
    .map((server) => server.trim())
    .filter(Boolean);

  try {
    dns.setServers(dnsServers);
    logger.info("Configured DNS servers for MongoDB SRV lookups", {
      dnsServers,
    });
  } catch (error) {
    logger.warn("Failed to configure DNS servers for MongoDB", {
      error: error.message,
    });
  }
};

const connectDB = async (retries = MAX_RETRIES) => {
  try {
    configureDnsForSrv();

    const currentUri = process.env.MONGODB_URI;
    
    // Check if URI is a placeholder from Atlas
    if (currentUri.includes("<db_password>")) {
       throw new Error("Invalid MongoDB URI: Please replace <db_password> with your actual password in .env");
    }

    const conn = await mongoose.connect(currentUri, {
      // Connection pool settings
      maxPoolSize: CONNECTION_POOL_SIZE,
      minPoolSize: 5,
      maxIdleTimeMS: 45000,

      // Timeout and retry settings
      serverSelectionTimeoutMS: CONNECTION_TIMEOUT,
      socketTimeoutMS: 45000,
      connectTimeoutMS: CONNECTION_TIMEOUT,
      retryWrites: true,
      retryReads: true,

      // Network optimization
      family: 4, // Force IPv4 – avoids IPv6 DNS delays on many networks
    });

    logger.info("MongoDB Connected successfully", {
      host: conn.connection.host,
      dbName: conn.connection.db?.databaseName,
      isMemoryServer: !!mongodInstance
    });

    // Handle connection events
    mongoose.connection.on("error", (err) => {
      logger.error("MongoDB connection error", err);
    });

    mongoose.connection.on("disconnected", () => {
      logger.warn("MongoDB disconnected. Attempting to reconnect...");
    });

    mongoose.connection.on("reconnected", () => {
      logger.info("MongoDB reconnected successfully");
    });

    // Cleanup legacy index from older schema versions
    try {
      const usersCollection = mongoose.connection.db.collection("users");
      const indexes = await usersCollection.indexes();
      const hasLegacyUsernameIndex = indexes.some(
        (idx) => idx.name === "username_1",
      );
      if (hasLegacyUsernameIndex) {
        await usersCollection.dropIndex("username_1");
        logger.info("Dropped legacy users.username_1 index");
      }
    } catch (indexError) {
      // Ignore "ns does not exist" errors - collection is new/doesn't exist yet
      if (!indexError.message.includes("ns does not exist")) {
        logger.warn("Index cleanup warning", { error: indexError.message });
      }
    }
  } catch (error) {
    const isConnRefused = error.message.includes("ECONNREFUSED") || error.message.includes("Server selection timed out");
    
    if (isConnRefused) {
      logger.error(
        `MongoDB connection failed: ${error.message}. Is MongoDB running?`,
        { attempt: MAX_RETRIES - retries + 1 }
      );
    } else {
      logger.error(
        `MongoDB connection failed (attempt ${MAX_RETRIES - retries + 1}/${MAX_RETRIES})`,
        error,
      );
    }

    if (retries > 0) {
      logger.info(`Retrying in ${RETRY_DELAY / 1000}s...`, {
        retriesRemaining: retries,
      });
      await new Promise((resolve) => setTimeout(resolve, RETRY_DELAY));
      return connectDB(retries - 1);
    }

    // Fallback for development: Start an in-memory MongoDB instance
    if (process.env.NODE_ENV !== 'production' && !mongodInstance) {
      try {
        logger.warn("⚠️ Local MongoDB not found. Starting in-memory fallback for development...");
        mongodInstance = await MongoMemoryServer.create();
        const uri = mongodInstance.getUri();
        logger.info(`🚀 In-memory MongoDB started at: ${uri}`);
        process.env.MONGODB_URI = uri;
        return connectDB(1); // Try one last time with memory server
      } catch (memError) {
        logger.critical("Failed to start in-memory MongoDB fallback", memError);
      }
    }

    logger.critical("Max retries reached. Unable to connect to MongoDB. Please check your database connection.");
    process.exit(1);
  }
};

export default connectDB;
