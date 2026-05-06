import { useEffect, useRef, useState } from 'react';
import { motion, useScroll, useInView } from 'motion/react';
import svgPaths from "../imports/svg-t8qzwshtp2";
import imgBackArrow from "figma:asset/52818968d8daa03fd6667845118af470e20c71c7.png";
import imgFrame1 from "figma:asset/040a3c667dcc879d4156f33acf977391876eefaa.png";
import imgEllipse7 from "figma:asset/6419896497f6a56d5538fab553a707cfdbb8ac54.png";
import imgGitHub from "figma:asset/b33fabf436a5add3adc9471b61e77a27cc3a741d.png";
import imgLinkedIn from "figma:asset/a6902ab50c505ea54bdd6d98c85549dcaa54f1ab.png";
import imgFacebook from "figma:asset/b5d2ff75cc5fe7d37bbda318ea52c1f4b9f334c9.png";
import imgEllipse8 from "figma:asset/c81ab8b1241945637f39b088bfded4f772b88485.png";
import imgRectangle from "figma:asset/587662697195b65a695aa3368db4072ce48f9588.png";

export default function App() {
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

      <HeroSection scrollToSection={scrollToSection} />
      <HowToUseSection scrollToSection={scrollToSection} />
      <MeetTheTeamSection scrollToSection={scrollToSection} />
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
      </div>
    </motion.nav>
  );
}

function HeroSection({ scrollToSection }: { scrollToSection: (id: string) => void }) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: false, amount: 0.3 });

  return (
    <section id="hero" ref={ref} className="relative min-h-screen bg-gradient-to-br from-[#1E0368] via-[#2D0A5E] to-[#1E0368] overflow-hidden flex items-center justify-center pt-20">
      <div className="relative z-10 max-w-7xl mx-auto px-6 py-20">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={isInView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.8 }}
          >
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="font-['Climate_Crisis'] text-7xl md:text-9xl text-white mb-8 leading-none"
            >
              TOX AGENT.
            </motion.h1>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.8, delay: 0.4 }}
              className="mb-8"
            >
              <p className="font-['Cal_Sans'] text-8xl md:text-9xl font-bold bg-gradient-to-r from-white to-purple-300 bg-clip-text text-transparent mb-2">
                100+
              </p>
              <p className="font-['Cal_Sans'] text-2xl text-white/90">
                users in first released version.
              </p>
            </motion.div>

            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.8, delay: 0.6 }}
              className="font-['Cal_Sans'] text-xl md:text-2xl text-white/90 mb-10 max-w-2xl"
            >
              From SMILES to structured insight - let AI reason through toxicity so you can focus on discovery.
            </motion.p>

            <motion.button
              initial={{ opacity: 0, scale: 0.9 }}
              animate={isInView ? { opacity: 1, scale: 1 } : {}}
              transition={{ duration: 0.5, delay: 0.8 }}
              whileHover={{ scale: 1.05, boxShadow: "0 0 40px rgba(255,255,255,0.3)" }}
              whileTap={{ scale: 0.95 }}
              onClick={() => scrollToSection('how-to-use')}
              className="group bg-transparent border-4 border-white rounded-full px-10 py-4 font-['Cal_Sans'] text-xl md:text-2xl text-white flex items-center gap-4 transition-all"
            >
              <span>GET STARTED</span>
              <motion.img
                whileHover={{ rotate: 360 }}
                transition={{ duration: 0.5 }}
                src={imgBackArrow}
                alt=""
                className="w-8 h-8 transform -scale-y-100 rotate-180"
              />
            </motion.button>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={isInView ? { opacity: 1, scale: 1 } : {}}
            transition={{ duration: 1, delay: 0.5 }}
            className="relative flex justify-center"
          >
            <div className="relative w-full max-w-md">
              <img src={imgFrame1} alt="TOX AGENT Platform" className="w-full rounded-3xl shadow-2xl" />
            </div>
          </motion.div>
        </div>

        <motion.p
          initial={{ opacity: 0 }}
          animate={isInView ? { opacity: 1 } : {}}
          transition={{ duration: 1, delay: 1 }}
          className="absolute bottom-10 right-10 font-['Orbitron'] text-2xl md:text-4xl text-white/50 font-bold"
        >
          PREDICT. INTERPRET. INNOVATE.
        </motion.p>
      </div>
    </section>
  );
}

function HowToUseSection({ scrollToSection }: { scrollToSection: (id: string) => void }) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: false, amount: 0.3 });

  return (
    <section id="how-to-use" ref={ref} className="relative min-h-screen bg-white overflow-hidden flex items-center justify-center py-20">
      <div className="absolute inset-0 bg-gradient-to-br from-[#1E0368] via-[#2D0A5E] to-[#1E0368] opacity-95" />

      <div className="relative z-10 max-w-7xl mx-auto px-6">
        <div className="text-center mb-12">
          <motion.h2
            initial={{ opacity: 0, y: -30 }}
            animate={isInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8 }}
            className="font-['Climate_Crisis'] text-6xl md:text-8xl text-white mb-4"
          >
            HOW TO USE
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: -20 }}
            animate={isInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="font-['Cal_Sans'] text-5xl md:text-7xl text-white"
          >
            TOX AGENT.
          </motion.p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={isInView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="relative"
          >
            <img src={imgRectangle} alt="How to use" className="w-full rounded-3xl shadow-2xl" />
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={isInView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="backdrop-blur-md bg-white/10 border-4 border-white rounded-3xl p-8 md:p-12 min-h-[500px] flex flex-col justify-between"
          >
            <div>
              <p className="font-['Cal_Sans'] text-7xl md:text-8xl font-bold bg-gradient-to-r from-white to-purple-300 bg-clip-text text-transparent mb-4">
                &lt;0.5s
              </p>
              <p className="font-['Cal_Sans'] text-4xl md:text-5xl text-white mb-8">
                per molecule
              </p>
            </div>

            <div>
              <p className="font-['Cal_Sans'] text-xl md:text-2xl text-white mb-8">
                Revolutionizing Molecular Toxicity Prediction with AI.
              </p>

              <motion.button
                whileHover={{ scale: 1.05, boxShadow: "0 0 30px rgba(255,255,255,0.3)" }}
                whileTap={{ scale: 0.95 }}
                onClick={() => window.open('/docs', '_blank')}
                className="bg-transparent border-4 border-white rounded-full px-8 py-3 font-['Cal_Sans'] text-xl md:text-2xl text-white flex items-center gap-4"
              >
                LEARN MORE
                <motion.img
                  whileHover={{ rotate: 360 }}
                  transition={{ duration: 0.5 }}
                  src={imgBackArrow}
                  alt=""
                  className="w-8 h-8 transform -scale-y-100 rotate-180"
                />
              </motion.button>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}

function MeetTheTeamSection({ scrollToSection }: { scrollToSection: (id: string) => void }) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: false, amount: 0.3 });

  const teamMembers = [
    {
      name: 'Nhat Minh',
      role: 'GNN Engineer',
      subRole: 'API Developer for Tool Calling',
      image: imgEllipse7
    },
    {
      name: 'Minh Le',
      role: 'Agentic AI Engineer',
      subRole: 'API Developer for Agents',
      image: imgEllipse8
    },
    {
      name: 'Alex Chen',
      role: 'ML Research Scientist',
      subRole: 'Model Architecture & Training',
      image: imgEllipse7
    },
    {
      name: 'Sarah Kim',
      role: 'Computational Biologist',
      subRole: 'Toxicology Domain Expert',
      image: imgEllipse8
    },
    {
      name: 'David Nguyen',
      role: 'Data Scientist',
      subRole: 'Chemical Informatics Specialist',
      image: imgEllipse7
    }
  ];

  return (
    <section id="team" ref={ref} className="relative min-h-screen bg-gradient-to-br from-[#1E0368] via-[#2D0A5E] to-[#1E0368] overflow-hidden flex items-center justify-center py-32">
      <div className="relative z-10 w-full max-w-[1800px] mx-auto px-8 md:px-12 lg:px-16">
        <motion.div
          initial={{ opacity: 0, y: -50 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
          className="text-center mb-20"
        >
          <h2 className="font-['Climate_Crisis'] text-6xl md:text-8xl text-white mb-4">MEET THE</h2>
          <p className="font-['Cal_Sans'] text-4xl md:text-6xl text-white">BIO RESEARCH TEAM.</p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-6 mb-16">
          {teamMembers.map((member, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 50 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.8, delay: index * 0.1 }}
              whileHover={{ y: -10, scale: 1.02 }}
              className="bg-black rounded-[40px] p-5 shadow-2xl relative overflow-hidden group"
            >
              <div className="relative z-10">
                <div className="w-full h-40 mb-3 rounded-3xl overflow-hidden">
                  <img src={member.image} alt={member.name} className="w-full h-full object-cover" />
                </div>

                <h3 className="font-['Cal_Sans'] text-lg text-white text-center mb-2">{member.name}</h3>
                <div className="text-center text-white/80 space-y-1 mb-3">
                  <p className="font-['Cal_Sans'] text-sm">{member.role}</p>
                  <p className="font-['Cal_Sans'] text-xs">{member.subRole}</p>
                </div>

                <div className="flex justify-center gap-3">
                  {[
                    { icon: imgGitHub, link: 'https://github.com' },
                    { icon: imgLinkedIn, link: 'https://linkedin.com' },
                    { icon: imgFacebook, link: 'https://facebook.com' }
                  ].map((social, i) => (
                    <motion.a
                      key={i}
                      href={social.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      whileHover={{ scale: 1.2, rotate: 5 }}
                      whileTap={{ scale: 0.9 }}
                      className="w-7 h-7"
                    >
                      <img src={social.icon} alt="" className="w-full h-full" />
                    </motion.a>
                  ))}
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        <div className="text-center space-y-3">
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={isInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="font-['Cal_Sans'] text-lg text-white/80"
          >
            /college of technology, national economics university
          </motion.p>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={isInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.7 }}
            className="font-['Cal_Sans'] text-lg text-white/80"
          >
            /faculty of data science &amp; artificial intelligence
          </motion.p>
        </div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={isInView ? { opacity: 1, scale: 1 } : {}}
          transition={{ duration: 0.5, delay: 0.8 }}
          className="flex justify-center mt-12"
        >
          <motion.button
            whileHover={{ scale: 1.05, boxShadow: "0 0 30px rgba(255,255,255,0.3)" }}
            whileTap={{ scale: 0.95 }}
            onClick={() => window.open('/team', '_blank')}
            className="bg-transparent border-4 border-white rounded-full px-10 py-3 font-['Cal_Sans'] text-xl text-white flex items-center gap-4"
          >
            ABOUT US
            <motion.img
              whileHover={{ rotate: 360 }}
              transition={{ duration: 0.5 }}
              src={imgBackArrow}
              alt=""
              className="w-8 h-8 transform -scale-y-100 rotate-180"
            />
          </motion.button>
        </motion.div>
      </div>
    </section>
  );
}

function TechStackSection() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: false, amount: 0.3 });

  const techLogos = [
    { name: 'Python', svg: svgPaths.p2c56b100 },
    { name: 'npm', svg: null },
    { name: 'Facebook', svg: null },
    { name: 'PostgreSQL', svg: svgPaths.p3c2e3680 },
    { name: 'React', svg: null },
    { name: 'Node.js', svg: svgPaths.p24206080 },
    { name: 'Figma', svg: svgPaths.pcccff00 },
    { name: 'Java', svg: null }
  ];

  return (
    <section id="tech-stack" ref={ref} className="relative min-h-screen bg-white overflow-hidden flex items-center justify-center py-20">
      <div className="relative z-10 w-full px-6">
        <motion.div
          initial={{ opacity: 0, y: -50 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
          className="text-center mb-16"
        >
          <h2 className="font-['Climate_Crisis'] text-6xl md:text-8xl text-[#1E0368] mb-4">ABOUT OUR</h2>
          <p className="font-['Cal_Sans'] text-4xl md:text-6xl font-bold bg-gradient-to-b from-[#1E0368] to-purple-600 bg-clip-text text-transparent">
            TECH STACKS
          </p>
        </motion.div>

        <div className="relative overflow-hidden mb-12">
          <motion.div
            animate={{ x: [0, -1400] }}
            transition={{
              duration: 30,
              repeat: Infinity,
              ease: "linear",
              repeatType: "loop"
            }}
            className="flex gap-8 md:gap-12 items-center whitespace-nowrap"
          >
            {[...techLogos, ...techLogos, ...techLogos, ...techLogos].map((tech, index) => (
              <motion.div
                key={index}
                whileHover={{ scale: 1.2, rotate: 5 }}
                className="w-24 h-24 md:w-32 md:h-32 flex-shrink-0 flex items-center justify-center"
              >
                <div className="w-full h-full rounded-2xl bg-white shadow-lg flex items-center justify-center border-2 border-[#1E0368]/10 hover:border-[#1E0368]/30 transition-all">
                  <span className="font-['Cal_Sans'] text-sm md:text-base text-[#1E0368] text-center px-2 font-semibold">{tech.name}</span>
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="text-center font-['Orbitron'] text-3xl md:text-4xl text-[#1E0368] font-bold"
        >
          PREDICT. INTERPRET. INNOVATE.
        </motion.p>
      </div>
    </section>
  );
}

function ImpressiveScaleSection() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: false, amount: 0.3 });

  const stats = [
    { value: '10,000+', label: 'TRAIN ON', sublabel: 'compounds' },
    { value: '50,000+', label: 'INDEXED', sublabel: 'scientific paper for MolRAG' },
    { value: '95%', label: 'COMPATIBLE WITH', sublabel: 'drug-like molecules' },
    { value: '70%', label: 'REDUCE', sublabel: 'early-stage screening costs' }
  ];

  return (
    <section id="impressive-scale" ref={ref} className="relative min-h-screen bg-gradient-to-br from-[#1E0368] via-[#2D0A5E] to-[#1E0368] overflow-hidden flex items-center justify-center py-20">
      <div className="relative z-10 max-w-7xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: -50 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
          className="text-center mb-16"
        >
          <h2 className="font-['Climate_Crisis'] text-6xl md:text-8xl text-white mb-4">IMPRESSIVE</h2>
          <p className="font-['Cal_Sans'] text-4xl md:text-6xl text-white">SCALE</p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-12 md:gap-16">
          {stats.map((stat, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={isInView ? { opacity: 1, scale: 1 } : {}}
              transition={{ duration: 0.8, delay: index * 0.1 }}
              whileHover={{ scale: 1.05 }}
              className="text-center"
            >
              <p className="font-['Cal_Sans'] text-xl md:text-2xl text-white/80 mb-3">{stat.label}</p>
              <p className="font-['Cal_Sans'] text-6xl md:text-8xl font-bold bg-gradient-to-r from-white to-purple-300 bg-clip-text text-transparent mb-3">
                {stat.value}
              </p>
              <p className="font-['Cal_Sans'] text-2xl md:text-4xl text-white">{stat.sublabel}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

function ResearchSourceSection() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: false, amount: 0.3 });

  const sources = [
    { name: 'HERG', link: '/research/herg' },
    { name: 'TOX21', link: '/research/tox21' }
  ];

  return (
    <section id="research-source" ref={ref} className="relative min-h-screen bg-white overflow-hidden flex items-center justify-center py-20">
      <div className="relative z-10 max-w-7xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: -50 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
          className="text-center mb-16"
        >
          <h2 className="font-['Climate_Crisis'] text-6xl md:text-8xl text-[#1E0368]">RESEARCH SOURCE</h2>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-5xl mx-auto">
          {sources.map((source, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 50 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.8, delay: index * 0.2 }}
              whileHover={{ y: -10, boxShadow: "0 30px 80px rgba(26,0,77,0.3)" }}
              className="border-4 border-[#1E0368] rounded-[60px] p-12 md:p-16 flex flex-col items-center justify-center min-h-[350px] relative overflow-hidden group cursor-pointer bg-white"
              onClick={() => window.open(source.link, '_blank')}
            >
              <h3 className="font-['Cal_Sans'] text-7xl md:text-9xl font-bold bg-gradient-to-r from-[#1E0368] to-purple-600 bg-clip-text text-transparent mb-8 relative z-10">
                {source.name}
              </h3>

              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                className="border-2 border-[#1E0368] rounded-full px-10 py-4 font-['Cal_Sans'] text-2xl md:text-3xl text-[#1E0368] relative z-10 transition-all hover:bg-[#1E0368] hover:text-white"
              >
                DISCOVER
              </motion.button>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

function CaseStudySection() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: false, amount: 0.3 });

  return (
    <section id="case-study" ref={ref} className="relative min-h-screen bg-gradient-to-br from-[#1E0368] via-[#2D0A5E] to-[#1E0368] overflow-hidden flex items-center justify-center py-20">
      <div className="relative z-10 max-w-7xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: -50 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
          className="text-center mb-16"
        >
          <h2 className="font-['Climate_Crisis'] text-6xl md:text-8xl text-white mb-8">CASE STUDY</h2>
          <p className="font-['Cal_Sans'] text-2xl md:text-3xl text-white max-w-4xl mx-auto">
            to empower your bio-informatics research?
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={isInView ? { opacity: 1, scale: 1 } : {}}
          transition={{ duration: 0.8, delay: 0.4 }}
          whileHover={{ scale: 1.02 }}
          className="bg-white rounded-3xl p-10 md:p-16 max-w-3xl mx-auto shadow-2xl cursor-pointer"
          onClick={() => window.open('/case-studies', '_blank')}
        >
          <motion.h3
            initial={{ opacity: 0, y: 20 }}
            animate={isInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="font-['Climate_Crisis'] text-5xl md:text-6xl text-[#1E0368] text-center mb-6"
          >
            aspirin
          </motion.h3>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={isInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.8 }}
            className="font-['Cal_Sans'] text-2xl md:text-3xl text-[#1E0368] text-center mb-12"
          >
            CC(=O)Oc1ccccc1C(=O)O
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={isInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8, delay: 1 }}
            className="text-center"
          >
            <span className="inline-block bg-green-500 text-white font-['Cal_Sans'] text-xl md:text-2xl px-10 py-4 rounded-full shadow-lg">
              Label: NON_TOXIC
            </span>
          </motion.div>
        </motion.div>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, delay: 1.2 }}
          className="text-center font-['Cal_Sans'] text-xl md:text-2xl text-white mt-12 max-w-4xl mx-auto"
        >
          Join the NEU Bio Research Team in pushing the boundaries of AI-driven drug safety.
        </motion.p>
      </div>
    </section>
  );
}

function CollaboratorsSection() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: false, amount: 0.3 });

  return (
    <section id="collaborators" ref={ref} className="relative min-h-[60vh] bg-white overflow-hidden flex items-center justify-center py-20">
      <div className="relative z-10 max-w-7xl mx-auto px-6 text-center">
        <motion.h2
          initial={{ opacity: 0, y: -50 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
          className="font-['Climate_Crisis'] text-5xl md:text-7xl text-[#1E0368] mb-12"
        >
          OUR COLLABORATORS
        </motion.h2>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={isInView ? { opacity: 1, scale: 1 } : {}}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="flex flex-wrap justify-center gap-8 md:gap-12"
        >
          {[1, 2, 3].map((i) => (
            <motion.div
              key={i}
              whileHover={{ scale: 1.1, rotate: 5 }}
              className="w-40 h-40 md:w-48 md:h-48 bg-gradient-to-br from-[#1E0368] to-purple-600 rounded-full shadow-xl cursor-pointer"
            />
          ))}
        </motion.div>
      </div>
    </section>
  );
}

function AffiliationSection() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: false, amount: 0.3 });

  const affiliations = ['FPT', 'VINAI', 'FDA', 'Twendee', 'FPT AI FACTORY'];

  return (
    <section id="affiliation" ref={ref} className="relative min-h-[60vh] bg-gradient-to-br from-[#1E0368] via-[#2D0A5E] to-[#1E0368] overflow-hidden flex items-center justify-center py-20">
      <div className="relative z-10 max-w-7xl mx-auto px-6 text-center">
        <motion.h2
          initial={{ opacity: 0, y: -50 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
          className="font-['Climate_Crisis'] text-5xl md:text-7xl text-white mb-12"
        >
          AFFILIATION THANKS TO
        </motion.h2>

        <motion.div
          initial={{ opacity: 0 }}
          animate={isInView ? { opacity: 1 } : {}}
          transition={{ duration: 1 }}
          className="flex flex-wrap justify-center gap-6 md:gap-8"
        >
          {affiliations.map((org, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 50 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.8, delay: index * 0.1 }}
              whileHover={{ scale: 1.1, y: -5 }}
              className="w-28 h-28 md:w-36 md:h-36 bg-white rounded-2xl flex items-center justify-center shadow-lg cursor-pointer"
            >
              <span className="font-['Cal_Sans'] text-xs md:text-sm text-[#1E0368] text-center px-2 font-bold">{org}</span>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}

function Footer() {
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
