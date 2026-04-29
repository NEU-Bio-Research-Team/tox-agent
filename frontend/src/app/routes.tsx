import { createBrowserRouter } from 'react-router';
import { RouteErrorBoundary } from './components/route-error-boundary';

export const router = createBrowserRouter([
  {
    path: '/',
    errorElement: <RouteErrorBoundary />,
    lazy: async () => {
      const module = await import('./pages/landing-page');
      return { Component: module.LandingPage };
    },
  },
  {
    path: '/analyze',
    errorElement: <RouteErrorBoundary />,
    lazy: async () => {
      const module = await import('./pages/index-page');
      return { Component: module.IndexPage };
    },
  },
  {
    path: '/report',
    errorElement: <RouteErrorBoundary />,
    lazy: async () => {
      const module = await import('./pages/report-page');
      return { Component: module.ReportPage };
    },
  },
  {
    path: '/chat',
    errorElement: <RouteErrorBoundary />,
    lazy: async () => {
      const module = await import('./pages/chatbot-page');
      return { Component: module.ChatbotPage };
    },
  },
  {
    path: '/settings',
    errorElement: <RouteErrorBoundary />,
    lazy: async () => {
      const module = await import('./pages/settings-page');
      return { Component: module.SettingsPage };
    },
  },
  {
    path: '/about',
    errorElement: <RouteErrorBoundary />,
    lazy: async () => {
      const module = await import('./pages/about-page');
      return { Component: module.AboutPage };
    },
  },
  {
    path: '/documents',
    errorElement: <RouteErrorBoundary />,
    lazy: async () => {
      const module = await import('./pages/documents-page');
      return { Component: module.DocumentsPage };
    },
  },
  {
    path: '/login',
    errorElement: <RouteErrorBoundary />,
    lazy: async () => {
      const module = await import('./pages/login-page');
      return { Component: module.LoginPage };
    },
  },
  {
    path: '/register',
    errorElement: <RouteErrorBoundary />,
    lazy: async () => {
      const module = await import('./pages/register-page');
      return { Component: module.RegisterPage };
    },
  },
]);
