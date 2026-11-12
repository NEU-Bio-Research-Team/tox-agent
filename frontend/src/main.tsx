import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './styles/index.css';
import { initTheme } from './shared/lib/theme';
import { firebaseAuthConfig, startSessionRefresh } from './features/auth/firebaseAuth';

initTheme();

const firebase = firebaseAuthConfig();
if (firebase) startSessionRefresh(firebase);

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
