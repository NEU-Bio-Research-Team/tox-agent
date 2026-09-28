import { useRef } from 'react';
import { motion, useInView } from 'motion/react';
import svgPaths from "../assets/svg-t8qzwshtp2";

export function TechStackSection() {
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
