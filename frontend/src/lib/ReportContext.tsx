import { createContext, useContext, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import type { AgentAnalyzeResponse } from './api';
import {
  getDefaultPreferences,
  loadUserPreferences,
  saveUserPreferences,
  type UserPreferences,
} from './user-preferences';

const REPORT_SESSION_KEY = 'tox_agent_report_snapshot';

function saveReportToSession(report: AgentAnalyzeResponse | null): void {
  try {
    if (report) {
      sessionStorage.setItem(REPORT_SESSION_KEY, JSON.stringify(report));
    } else {
      sessionStorage.removeItem(REPORT_SESSION_KEY);
    }
  } catch {
    // sessionStorage may be unavailable (private browsing, quota exceeded)
  }
}

function loadReportFromSession(): AgentAnalyzeResponse | null {
  try {
    const raw = sessionStorage.getItem(REPORT_SESSION_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as AgentAnalyzeResponse;
  } catch {
    return null;
  }
}

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
  const [report, setReportState] = useState<AgentAnalyzeResponse | null>(() => loadReportFromSession());
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preferences, setPreferencesState] = useState<UserPreferences>(() => loadUserPreferences());

  const setReport = (nextReport: AgentAnalyzeResponse | null) => {
    setReportState(nextReport);
    saveReportToSession(nextReport);
  };

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
