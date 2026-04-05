import { useEffect, useState } from 'react';
import { RouterProvider } from 'react-router';
import { router } from './routes';
import { ReportProvider } from '../lib/ReportContext';
import { ReleaseNotesModal } from './components/release-notes-modal';
import { RELEASE_NOTES_EVENT, RELEASE_NOTES_STORAGE_KEY } from './release-notes';
import { AuthProvider } from './components/contexts/auth-context';

export default function App() {
  const [releaseNotesOpen, setReleaseNotesOpen] = useState(false);

  useEffect(() => {
    const handleOpen = () => {
      setReleaseNotesOpen(true);
    };

    window.addEventListener(RELEASE_NOTES_EVENT, handleOpen);
    return () => {
      window.removeEventListener(RELEASE_NOTES_EVENT, handleOpen);
    };
  }, []);

  useEffect(() => {
    try {
      const seen = window.localStorage.getItem(RELEASE_NOTES_STORAGE_KEY);
      if (!seen) {
        window.localStorage.setItem(RELEASE_NOTES_STORAGE_KEY, '1');
        setReleaseNotesOpen(true);
      }
    } catch {
      // Fallback to showing the release notes when localStorage is unavailable.
      setReleaseNotesOpen(true);
    }
  }, []);

  return (
    <AuthProvider>
      <ReportProvider>
        <RouterProvider router={router} />
        <ReleaseNotesModal open={releaseNotesOpen} onOpenChange={setReleaseNotesOpen} />
      </ReportProvider>
    </AuthProvider>
  );
}
