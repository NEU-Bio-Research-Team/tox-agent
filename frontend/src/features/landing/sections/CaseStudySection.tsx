import { useRef } from 'react';
import { motion, useInView } from 'motion/react';

export function CaseStudySection() {
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
          onClick={() => window.location.assign('/predict')}
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
