import { createContext, useContext, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import type { AgentAnalyzeResponse } from './api';
import {
  getDefaultPreferences,
  loadUserPreferences,
  saveUserPreferences,
  type UserPreferences,
} from './user-preferences';

interface ReportContextValue {
  report: AgentAnalyzeResponse | null;
  setReport: (nextReport: AgentAnalyzeResponse | null) => void;
  isLoading: boolean;
  setIsLoading: (next: boolean) => void;
  error: string | null;
  setError: (next: string | null) => void;
  preferences: UserPreferences;
  setPreferences: (next: UserPreferences) => void;
  resetPreferences: () => void;
}

const ReportContext = createContext<ReportContextValue | undefined>(undefined);

export function ReportProvider({ children }: { children: ReactNode }) {
  const [report, setReport] = useState<AgentAnalyzeResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preferences, setPreferencesState] = useState<UserPreferences>(() => loadUserPreferences());

  const setPreferences = (next: UserPreferences) => {
    const normalized = saveUserPreferences(next);
    setPreferencesState(normalized);
  };

  const resetPreferences = () => {
    const defaults = getDefaultPreferences();
    const normalized = saveUserPreferences(defaults);
    setPreferencesState(normalized);
  };

  const value = useMemo(
    () => ({
      report,
      setReport,
      isLoading,
      setIsLoading,
      error,
      setError,
      preferences,
      setPreferences,
      resetPreferences,
    }),
    [error, isLoading, preferences, report],
  );

  return <ReportContext.Provider value={value}>{children}</ReportContext.Provider>;
}

export function useReport() {
  const context = useContext(ReportContext);
  if (!context) {
    throw new Error('useReport must be used within a ReportProvider');
  }
  return context;
}
