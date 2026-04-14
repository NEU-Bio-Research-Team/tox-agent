import { createBrowserRouter } from 'react-router';

export const router = createBrowserRouter([
  {
    path: '/',
    lazy: async () => {
      const module = await import('./pages/index-page');
      return { Component: module.IndexPage };
    },
  },
  {
    path: '/report',
    lazy: async () => {
      const module = await import('./pages/report-page');
      return { Component: module.ReportPage };
    },
  },
  {
    path: '/chat',
    lazy: async () => {
      const module = await import('./pages/chatbot-page');
      return { Component: module.ChatbotPage };
    },
  },
  {
    path: '/settings',
    lazy: async () => {
      const module = await import('./pages/settings-page');
      return { Component: module.SettingsPage };
    },
  },
  {
    path: '/about',
    lazy: async () => {
      const module = await import('./pages/about-page');
      return { Component: module.AboutPage };
    },
  },
  {
    path: '/documents',
    lazy: async () => {
      const module = await import('./pages/documents-page');
      return { Component: module.DocumentsPage };
    },
  },
  {
    path: '/login',
    lazy: async () => {
      const module = await import('./pages/login-page');
      return { Component: module.LoginPage };
    },
  },
  {
    path: '/register',
    lazy: async () => {
      const module = await import('./pages/register-page');
      return { Component: module.RegisterPage };
    },
  },
]);
