import { useRef } from 'react';
import { motion, useInView } from 'motion/react';

export function AffiliationSection() {
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
