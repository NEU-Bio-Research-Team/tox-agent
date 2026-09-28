import { motion } from 'motion/react';

export function Footer() {
  return (
    <footer className="bg-[#0D0021] text-white py-12">
      <div className="max-w-7xl mx-auto px-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-12 mb-8">
          <div>
            <h3 className="font-['Climate_Crisis'] text-3xl mb-4">TOX AGENT</h3>
            <p className="font-['Cal_Sans'] text-sm opacity-80">
              Revolutionizing molecular toxicity prediction with AI
            </p>
          </div>

          <div>
            <h4 className="font-['Cal_Sans'] text-xl mb-4">Quick Links</h4>
            <ul className="space-y-2 font-['Cal_Sans'] text-sm opacity-80">
              <li><a href="/about" className="hover:opacity-100 transition-opacity">About Us</a></li>
              <li><a href="/research" className="hover:opacity-100 transition-opacity">Research</a></li>
              <li><a href="/docs" className="hover:opacity-100 transition-opacity">Documentation</a></li>
              <li><a href="/contact" className="hover:opacity-100 transition-opacity">Contact</a></li>
            </ul>
          </div>

          <div>
            <h4 className="font-['Cal_Sans'] text-xl mb-4">Connect With Us</h4>
            <div className="flex gap-4">
              <motion.a
                href="https://github.com"
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
          <p className="font-['Cal_Sans'] text-sm opacity-60">
            &copy; 2026 TOX AGENT. All rights reserved. | NEU Bio Research Team
          </p>
        </div>
      </div>
    </footer>
  );
}
