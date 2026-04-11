// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyCXFLnZUId6rz9SP2r4glvB624eUKiExfc",
  authDomain: "tox-agent.firebaseapp.com",
  projectId: "tox-agent",
  storageBucket: "tox-agent.firebasestorage.app",
  messagingSenderId: "147622422092",
  appId: "1:147622422092:web:4ee7a9c13fbf6a28b2b6e9",
  measurementId: "G-GE1TRH1M6V"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);