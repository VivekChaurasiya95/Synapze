import mongoose from "mongoose";
import dotenv from "dotenv";

dotenv.config();

const findAllEmails = async () => {
  try {
    await mongoose.connect(process.env.MONGODB_URI);
    console.log("Connected to MongoDB\n");

    const db = mongoose.connection.db;
    const userCollection = db.collection("users");

    // Get ALL documents without any filtering
    const allDocs = await userCollection.find({}).toArray();
    console.log(`Total documents in users collection: ${allDocs.length}\n`);

    allDocs.forEach((doc, i) => {
      console.log(`[${i + 1}] ID: ${doc._id}`);
      console.log(`    Email: "${doc.email}"`);
      console.log(
        `    Email (raw bytes): ${Buffer.from(doc.email).toString("hex")}`,
      );
      console.log(`    Name: ${doc.name}`);
      console.log(`    Role: ${doc.role}`);
      console.log("");
    });

    // Also check for any fields that might be used in a unique index
    console.log("\nChecking unique-like fields:");
    const docs = allDocs;
    if (docs[0]) {
      console.log("First document fields:");
      Object.keys(docs[0]).forEach((key) => {
        console.log(
          `  - ${key}: ${typeof docs[0][key]} = ${JSON.stringify(docs[0][key]).substring(0, 50)}`,
        );
      });
    }

    await mongoose.connection.close();
  } catch (error) {
    console.error("Error:", error.message);
    process.exit(1);
  }
};

findAllEmails();
