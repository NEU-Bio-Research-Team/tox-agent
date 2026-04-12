import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';

const firebaseConfig = {
  apiKey: 'AIzaSyCXFLnZUId6rz9SP2r4glvB624eUKiExfc',
  authDomain: 'tox-agent.firebaseapp.com',
  projectId: 'tox-agent',
  storageBucket: 'tox-agent.firebasestorage.app',
  messagingSenderId: '147622422092',
  appId: '1:147622422092:web:4ee7a9c13fbf6a28b2b6e9',
  measurementId: 'G-GE1TRH1M6V',
};

const app = initializeApp(firebaseConfig);

export const auth = getAuth(app);
export const db = getFirestore(app);
