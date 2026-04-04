import { RouterProvider } from 'react-router';
import { router } from './routes';
import { AuthProvider } from './components/contexts/auth-context';
import { AIChatbot } from './components/ai-chatbot';

export default function App() {
  return (
    <AuthProvider>
      <RouterProvider router={router} />
      <AIChatbot />
    </AuthProvider>
  );
}
