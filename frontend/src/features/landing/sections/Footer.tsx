import { motion } from 'motion/react';
import { FEEDBACK_URL, REPO_URL } from '../../../app/links';

export function Footer() {
  return (
    <footer className="bg-brand-night text-white py-12">
      <div className="max-w-7xl mx-auto px-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-12 mb-8">
          <div>
            <h3 className="font-sans font-extrabold tracking-tight text-3xl mb-4">TOX AGENT</h3>
            <p className="font-sans font-bold text-sm opacity-80">
              Revolutionizing molecular toxicity prediction with AI
            </p>
          </div>

          <div>
            <h4 className="font-sans font-bold text-xl mb-4">Quick Links</h4>
            <ul className="space-y-2 font-sans font-bold text-sm opacity-80">
              <li><a href="/docs/intro" className="hover:opacity-100 transition-opacity">Documentation</a></li>
              <li><a href="/docs/principles" className="hover:opacity-100 transition-opacity">Principles</a></li>
              <li><a href="/analyze" className="hover:opacity-100 transition-opacity">Quick Predict</a></li>
              <li><a href={FEEDBACK_URL} target="_blank" rel="noopener noreferrer" className="hover:opacity-100 transition-opacity">Feedback</a></li>
            </ul>
          </div>

          <div>
            <h4 className="font-sans font-bold text-xl mb-4">Connect With Us</h4>
            <div className="flex gap-4">
              <motion.a
                href={REPO_URL}
                target="_blank"
                rel="noopener noreferrer"
                whileHover={{ scale: 1.2 }}
                className="opacity-80 hover:opacity-100 transition-opacity"
              >
                GitHub
              </motion.a>
              <motion.a
                href="https://linkedin.com"
                target="_blank"
                rel="noopener noreferrer"
                whileHover={{ scale: 1.2 }}
                className="opacity-80 hover:opacity-100 transition-opacity"
              >
                LinkedIn
              </motion.a>
            </div>
          </div>
        </div>

        <div className="border-t border-white/20 pt-8 text-center">
          <p className="font-sans font-bold text-sm opacity-60">
            &copy; 2026 TOX AGENT. All rights reserved. | NEU Bio Research Team
          </p>
        </div>
      </div>
    </footer>
  );
}
