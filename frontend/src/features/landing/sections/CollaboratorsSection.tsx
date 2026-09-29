import { useRef } from 'react';
import { motion, useInView } from 'motion/react';

export function CollaboratorsSection() {
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
