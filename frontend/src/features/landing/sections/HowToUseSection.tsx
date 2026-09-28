import { useRef } from 'react';
import { motion, useInView } from 'motion/react';
import imgBackArrow from "../assets/52818968d8daa03fd6667845118af470e20c71c7.png";
import imgRectangle from "../assets/587662697195b65a695aa3368db4072ce48f9588.png";

export function HowToUseSection() {
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
                onClick={() => window.location.assign('/predict')}
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
