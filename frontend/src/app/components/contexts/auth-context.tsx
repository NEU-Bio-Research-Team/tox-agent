import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { initializeDemoAccount } from '../../utils/demo-account';

interface User {
  id: string;
  email: string;
  name: string;
}

interface AuthContextType {
  user: User | null;
  login: (email: string, password: string) => Promise<boolean>;
  register: (email: string, password: string, name: string) => Promise<boolean>;
  logout: () => void;
  isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    // Initialize demo account
    initializeDemoAccount();
    
    // Check for existing session
    const savedUser = localStorage.getItem('toxagent_user');
    if (savedUser) {
      setUser(JSON.parse(savedUser));
    }
  }, []);

  const register = async (email: string, password: string, name: string): Promise<boolean> => {
    try {
      // Get existing users
      const usersJson = localStorage.getItem('toxagent_users') || '[]';
      const users = JSON.parse(usersJson);
      
      // Check if user already exists
      if (users.find((u: any) => u.email === email)) {
        return false;
      }
      
      // Create new user
      const newUser = {
        id: crypto.randomUUID(),
        email,
        password, // In production, this would be hashed
        name,
      };
      
      users.push(newUser);
      localStorage.setItem('toxagent_users', JSON.stringify(users));
      
      // Auto-login after registration
      const userSession = { id: newUser.id, email: newUser.email, name: newUser.name };
      setUser(userSession);
      localStorage.setItem('toxagent_user', JSON.stringify(userSession));
      
      return true;
    } catch (error) {
      console.error('Registration error:', error);
      return false;
    }
  };

  const login = async (email: string, password: string): Promise<boolean> => {
    try {
      const usersJson = localStorage.getItem('toxagent_users') || '[]';
      const users = JSON.parse(usersJson);
      
      const foundUser = users.find((u: any) => u.email === email && u.password === password);
      
      if (foundUser) {
        const userSession = { id: foundUser.id, email: foundUser.email, name: foundUser.name };
        setUser(userSession);
        localStorage.setItem('toxagent_user', JSON.stringify(userSession));
        return true;
      }
      
      return false;
    } catch (error) {
      console.error('Login error:', error);
      return false;
    }
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('toxagent_user');
  };

  return (
    <AuthContext.Provider value={{ user, login, register, logout, isAuthenticated: !!user }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
