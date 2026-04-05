// Initialize demo account on first load
export function initializeDemoAccount() {
  const users = localStorage.getItem('toxagent_users');
  
  if (!users) {
    // Create demo account
    const demoUser = {
      id: 'demo-user-001',
      email: 'demo@toxagent.ai',
      password: 'demo123',
      name: 'Demo User',
    };
    
    localStorage.setItem('toxagent_users', JSON.stringify([demoUser]));
  }
}
