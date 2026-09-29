import { useRef } from 'react';
import { motion, useInView } from 'motion/react';
import imgBackArrow from "../assets/52818968d8daa03fd6667845118af470e20c71c7.png";
import imgFrame1 from "../assets/040a3c667dcc879d4156f33acf977391876eefaa.png";

export function HeroSection() {
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
              onClick={() => window.location.assign('/sessions')}
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
