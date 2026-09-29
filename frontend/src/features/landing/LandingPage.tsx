import { useEffect, useState } from 'react';
import { motion, useScroll } from 'motion/react';
import { HeroSection } from './sections/HeroSection';
import { HowToUseSection } from './sections/HowToUseSection';
import { MeetTheTeamSection } from './sections/MeetTheTeamSection';
import { TechStackSection } from './sections/TechStackSection';
import { ImpressiveScaleSection } from './sections/ImpressiveScaleSection';
import { ResearchSourceSection } from './sections/ResearchSourceSection';
import { CaseStudySection } from './sections/CaseStudySection';
import { CollaboratorsSection } from './sections/CollaboratorsSection';
import { AffiliationSection } from './sections/AffiliationSection';
import { Footer } from './sections/Footer';

export function LandingPage() {
  const { scrollYProgress } = useScroll();
  const [activeSection, setActiveSection] = useState('hero');

  useEffect(() => {
    document.documentElement.style.scrollBehavior = 'smooth';
  }, []);

  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
      setActiveSection(id);
    }
  };

  return (
    <div className="w-full bg-white overflow-x-hidden">
      <motion.div
        className="fixed top-0 left-0 right-0 h-1 bg-gradient-to-r from-[#1E0368] via-purple-600 to-[#1E0368] z-50 origin-left"
        style={{ scaleX: scrollYProgress }}
      />

      <Navigation activeSection={activeSection} scrollToSection={scrollToSection} />

      <HeroSection />
      <HowToUseSection />
      <MeetTheTeamSection />
      <TechStackSection />
      <ImpressiveScaleSection />
      <ResearchSourceSection />
      <CaseStudySection />
      <CollaboratorsSection />
      <AffiliationSection />

      <Footer />
    </div>
  );
}

function Navigation({ activeSection, scrollToSection }: { activeSection: string; scrollToSection: (id: string) => void }) {
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => setIsScrolled(window.scrollY > 50);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const navItems = [
    { id: 'hero', label: 'Home' },
    { id: 'how-to-use', label: 'How To Use' },
    { id: 'team', label: 'Team' },
    { id: 'tech-stack', label: 'Tech Stack' },
    { id: 'case-study', label: 'Case Study' }
  ];

  return (
    <motion.nav
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      className={`fixed top-0 left-0 right-0 z-40 transition-all duration-300 ${
        isScrolled ? 'bg-[#1E0368]/95 backdrop-blur-lg shadow-lg' : 'bg-transparent'
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        <motion.div
          whileHover={{ scale: 1.05 }}
          className="font-['Climate_Crisis'] text-3xl text-white cursor-pointer"
          onClick={() => scrollToSection('hero')}
        >
          TOX AGENT
        </motion.div>

        <div className="hidden md:flex gap-8">
          {navItems.map((item) => (
            <motion.button
              key={item.id}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => scrollToSection(item.id)}
              className={`font-['Cal_Sans'] text-lg transition-colors ${
                activeSection === item.id ? 'text-white' : 'text-white/70 hover:text-white'
              }`}
            >
              {item.label}
            </motion.button>
          ))}
        </div>
        <motion.button
          type="button"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => window.location.assign('/predict')}
          className="border-2 border-white rounded-full px-5 py-2 font-['Cal_Sans'] text-sm text-white transition-colors hover:bg-white hover:text-[#1E0368]"
        >
          QUICK PREDICT
        </motion.button>
      </div>
    </motion.nav>
  );
}
