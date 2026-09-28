import { useRef } from 'react';
import { motion, useInView } from 'motion/react';
import imgBackArrow from "../assets/52818968d8daa03fd6667845118af470e20c71c7.png";
import imgEllipse7 from "../assets/6419896497f6a56d5538fab553a707cfdbb8ac54.png";
import imgGitHub from "../assets/b33fabf436a5add3adc9471b61e77a27cc3a741d.png";
import imgLinkedIn from "../assets/a6902ab50c505ea54bdd6d98c85549dcaa54f1ab.png";
import imgFacebook from "../assets/b5d2ff75cc5fe7d37bbda318ea52c1f4b9f334c9.png";
import imgEllipse8 from "../assets/c81ab8b1241945637f39b088bfded4f772b88485.png";

export function MeetTheTeamSection() {
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
            onClick={() => window.location.assign('/about')}
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
