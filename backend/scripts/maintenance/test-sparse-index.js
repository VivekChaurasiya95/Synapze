import mongoose from "mongoose";
import dotenv from "dotenv";

dotenv.config();

const testSparseIndex = async () => {
  try {
    await mongoose.connect(process.env.MONGODB_URI);
    console.log("Connected to MongoDB\n");

    const db = mongoose.connection.db;
    const testCollection = db.collection("test_sparse");

    // Drop the collection if it exists
    try {
      await testCollection.drop();
      console.log("✓ Dropped test collection");
    } catch (e) {
      // Collection doesn't exist
    }

    // Create a sparse unique index on a field
    console.log("\nCreating sparse unique index...");
    await testCollection.createIndex(
      { sparseField: 1 },
      { unique: true, sparse: true },
    );
    console.log("✓ Created sparse unique index\n");

    // Insert first document with null value
    console.log("Inserting first document with null sparseField...");
    try {
      await testCollection.insertOne({
        name: "Doc1",
        sparseField: null,
      });
      console.log("✓ First document inserted\n");
    } catch (e) {
      console.log(`❌ Failed: ${e.message}\n`);
    }

    // Insert second document with null value
    console.log("Inserting second document with null sparseField...");
    try {
      await testCollection.insertOne({
        name: "Doc2",
        sparseField: null,
      });
      console.log("✓ Second document inserted\n");
    } catch (e) {
      console.log(`❌ Failed: ${e.message}\n`);
    }

    // Check documents
    const docs = await testCollection.find({}).toArray();
    console.log(`Total documents: ${docs.length}`);
    docs.forEach((d) =>
      console.log(`  - ${d.name}: sparseField=${d.sparseField}`),
    );

    // Clean up
    await testCollection.drop();
    await mongoose.connection.close();
  } catch (error) {
    console.error("Error:", error.message);
    process.exit(1);
  }
};

testSparseIndex();
