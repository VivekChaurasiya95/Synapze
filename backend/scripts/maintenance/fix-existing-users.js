import mongoose from "mongoose";
import dotenv from "dotenv";

dotenv.config();

const fixExistingUsers = async () => {
  try {
    await mongoose.connect(process.env.MONGODB_URI);
    console.log("Connected to MongoDB\n");

    const db = mongoose.connection.db;
    const userCollection = db.collection("users");

    // Update all documents with firebaseUid: null to remove the field
    console.log("Fixing existing users with null firebaseUid...");
    const result = await userCollection.updateMany(
      { firebaseUid: null },
      { $unset: { firebaseUid: "" } },
    );

    console.log(`✓ Updated ${result.modifiedCount} documents\n`);

    // Show remaining users
    const users = await userCollection.find({}).toArray();
    console.log(`Total users: ${users.length}`);
    users.forEach((u) => {
      console.log(
        `  - ${u.email}: firebaseUid=${u.firebaseUid || "(not set)"}`,
      );
    });

    await mongoose.connection.close();
  } catch (error) {
    console.error("Error:", error.message);
    process.exit(1);
  }
};

fixExistingUsers();
