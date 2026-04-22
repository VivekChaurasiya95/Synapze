import mongoose from "mongoose";
import dotenv from "dotenv";

dotenv.config();

const fixSparseIndex = async () => {
  try {
    await mongoose.connect(process.env.MONGODB_URI);
    console.log("Connected to MongoDB\n");

    const db = mongoose.connection.db;
    const userCollection = db.collection("users");

    // Drop the firebaseUid index
    try {
      await userCollection.dropIndex("firebaseUid_1");
      console.log("✓ Dropped firebaseUid_1 index");
    } catch (e) {
      console.log(`- Could not drop firebaseUid_1: ${e.message}`);
    }

    // Recreate it with SPARSE option (so null values don't cause duplicates)
    console.log("\nCreating firebaseUid sparse unique index...");
    try {
      await userCollection.createIndex(
        { firebaseUid: 1 },
        { unique: true, sparse: true },
      );
      console.log("✓ Created firebaseUid index with sparse: true\n");
    } catch (e) {
      console.log(`❌ Failed: ${e.message}`);
    }

    // List all indexes
    console.log("Final indexes:");
    const indexes = await userCollection.listIndexes().toArray();
    indexes.forEach((idx) => {
      const parts = [`${idx.name}: ${JSON.stringify(idx.key)}`];
      if (idx.unique) parts.push("[UNIQUE]");
      if (idx.sparse) parts.push("[SPARSE]");
      console.log(`  - ${parts.join(" ")}`);
    });

    await mongoose.connection.close();
  } catch (error) {
    console.error("Error:", error.message);
    process.exit(1);
  }
};

fixSparseIndex();
