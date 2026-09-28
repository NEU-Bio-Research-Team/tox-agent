import { useRef } from 'react';
import { motion, useInView } from 'motion/react';

export function ImpressiveScaleSection() {
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
