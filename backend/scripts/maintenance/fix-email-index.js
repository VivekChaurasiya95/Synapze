import mongoose from "mongoose";
import dotenv from "dotenv";

dotenv.config();

const fixEmailIndex = async () => {
  try {
    await mongoose.connect(process.env.MONGODB_URI);
    console.log("Connected to MongoDB\n");

    const db = mongoose.connection.db;
    const userCollection = db.collection("users");

    // Get all current indexes
    console.log("Current indexes:");
    const indexes = await userCollection.listIndexes().toArray();
    indexes.forEach((idx) => {
      console.log(`  - ${idx.name}: ${JSON.stringify(idx.key)}`);
      if (idx.name && idx.name.includes("email")) {
        console.log(`    Options: ${JSON.stringify(idx)}`);
      }
    });

    // Drop the email index if it exists
    try {
      await userCollection.dropIndex("email_1");
      console.log("\n✓ Dropped email_1 index");
    } catch (e) {
      console.log(`\n- No email_1 index to drop: ${e.message}`);
    }

    // Drop unique email index if it exists
    try {
      await userCollection.dropIndex("email_1_unique");
      console.log("✓ Dropped email_1_unique index");
    } catch (e) {
      console.log(`- No email_1_unique index to drop`);
    }

    // Try any index with email
    for (const idx of indexes) {
      if (idx.key.email === 1 && idx.name !== "_id_") {
        try {
          await userCollection.dropIndex(idx.name);
          console.log(`✓ Dropped ${idx.name} index`);
        } catch (e) {
          console.log(`- Could not drop ${idx.name}`);
        }
      }
    }

    console.log("\n--- Recreating User schema with Mongoose ---");

    // Force Mongoose to recreate indexes
    const User = await import("./src/models/User.js").then((m) => m.default);

    // Ensure indexes are created
    await User.collection.dropIndexes();
    console.log("✓ Dropped all indexes via Mongoose");

    await User.syncIndexes();
    console.log("✓ Synced Mongoose indexes");

    // List new indexes
    const newIndexes = await userCollection.listIndexes().toArray();
    console.log("\nNew indexes:");
    newIndexes.forEach((idx) => {
      console.log(`  - ${idx.name}: ${JSON.stringify(idx.key)}`);
    });

    console.log("\n✅ Email index fixed!");

    await mongoose.connection.close();
  } catch (error) {
    console.error("Error:", error.message);
    process.exit(1);
  }
};

fixEmailIndex();
