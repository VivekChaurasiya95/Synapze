import mongoose from "mongoose";
import dotenv from "dotenv";

dotenv.config();

const properlyFixIndex = async () => {
  try {
    await mongoose.connect(process.env.MONGODB_URI);
    console.log("Connected to MongoDB\n");

    const db = mongoose.connection.db;
    const userCollection = db.collection("users");

    // Drop ALL indexes except _id_
    try {
      await userCollection.dropIndexes();
      console.log("✓ Dropped all indexes");
    } catch (e) {
      console.log(`- Could not drop indexes: ${e.message}`);
    }

    // Check for duplicate emails (this might be the real issue)
    const emailCounts = {};
    const allDocs = await userCollection.find({}).toArray();

    console.log(
      `\nChecking all ${allDocs.length} documents for duplicate emails...`,
    );
    let duplicates = 0;
    for (const doc of allDocs) {
      const email = doc.email.toLowerCase();
      if (emailCounts[email]) {
        console.log(
          `  ⚠️  DUPLICATE: ${email} (IDs: ${emailCounts[email].join(", ")} and ${doc._id})`,
        );
        duplicates++;
      } else {
        emailCounts[email] = [doc._id.toString()];
      }
    }

    if (duplicates > 0) {
      console.log(
        `\nFound ${duplicates} duplicate emails. These need to be cleaned up manually.`,
      );
    } else {
      console.log("\n✓ No duplicate emails found");
    }

    // Recreate the unique index
    console.log("\nCreating unique email index...");
    try {
      await userCollection.createIndex({ email: 1 }, { unique: true });
      console.log("✓ Created unique email index");
    } catch (e) {
      console.log(`❌ Failed to create unique index: ${e.message}`);
      if (e.message.includes("duplicate")) {
        console.log("   Reason: Duplicate email values exist in collection");
      }
    }

    // List current indexes
    console.log("\nCurrent indexes:");
    const indexes = await userCollection.listIndexes().toArray();
    indexes.forEach((idx) => {
      const unique = idx.unique ? " [UNIQUE]" : "";
      console.log(`  - ${idx.name}: ${JSON.stringify(idx.key)}${unique}`);
    });

    await mongoose.connection.close();
  } catch (error) {
    console.error("Error:", error.message);
    process.exit(1);
  }
};

properlyFixIndex();
