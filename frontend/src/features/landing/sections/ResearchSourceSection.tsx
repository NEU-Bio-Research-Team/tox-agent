import { useRef } from 'react';
import { motion, useInView } from 'motion/react';

export function ResearchSourceSection() {
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
