import mongoose from "mongoose";
import dotenv from "dotenv";
import User from "./src/models/User.js";

dotenv.config();

const fixAllIndexes = async () => {
  try {
    await mongoose.connect(process.env.MONGODB_URI);
    console.log("Connected to MongoDB\n");

    // Use Mongoose to ensure all indexes from schema are created
    console.log("Syncing all Mongoose indexes...");
    await User.syncIndexes();
    console.log("✓ Indexes synced\n");

    // List all indexes
    const db = mongoose.connection.db;
    const userCollection = db.collection("users");

    console.log("Final indexes:");
    const indexes = await userCollection.listIndexes().toArray();
    indexes.forEach((idx) => {
      const unique = idx.unique ? " [UNIQUE]" : "";
      const sparse = idx.sparse ? " [SPARSE]" : "";
      console.log(
        `  - ${idx.name}: ${JSON.stringify(idx.key)}${unique}${sparse}`,
      );
    });

    await mongoose.connection.close();
  } catch (error) {
    console.error("Error:", error.message);
    process.exit(1);
  }
};

fixAllIndexes();
