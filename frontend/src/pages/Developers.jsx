import { useState, useEffect, useRef } from "react";
import {
  Github,
  Linkedin,
  Database,
  Bug,
  Mail,
  Globe,
  Layers,
  Code2,
  Sparkles,
  ExternalLink,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

/* -- Scroll-reveal hook -- */
const useReveal = (threshold = 0.15) => {
  const ref = useRef(null);
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          setVisible(true);
          obs.unobserve(el);
        }
      },
      { threshold },
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [threshold]);
  return [ref, visible];
};

const RevealSection = ({ children, className = "", delay = 0 }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-100px" }}
      transition={{ duration: 0.8, delay: delay / 1000, ease: "easeOut" }}
      className={className}
    >
      {children}
    </motion.div>
  );
};

const DeveloperCard = ({ member, idx, primaryCat, gradient }) => {
  const CatIcon = {
    Frontend: Layers,
    Backend: Database,
    Testing: Bug,
  };

  const badgeColors = {
    Frontend: "bg-blue-500/10 text-blue-400 border-blue-500/20",
    Backend: "bg-blue-500/10 text-blue-400 border-blue-500/20",
    Testing: "bg-blue-500/10 text-blue-400 border-blue-500/20",
  };

  return (
    <motion.div
      layout
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
      transition={{ duration: 0.5, delay: idx * 0.1 }}
      whileHover={{ y: -10 }}
      className="group relative h-full"
    >
      {/* Glow Backdrop */}
      <div className={`absolute -inset-0.5 bg-gradient-to-r ${gradient} rounded-[2rem] opacity-0 group-hover:opacity-30 blur-xl transition-all duration-700`} />

      <div className="relative h-full bg-brand-surface/80 backdrop-blur-xl rounded-[1.8rem] border border-brand-border/50 group-hover:border-brand-accent/40 overflow-hidden transition-all duration-500 flex flex-col">
        {/* Shine Effect */}
        <div className="absolute inset-0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000 bg-gradient-to-r from-transparent via-white/5 to-transparent pointer-events-none" />

        {/* Header Background */}
        <div className={`h-32 bg-gradient-to-br ${gradient} opacity-10 group-hover:opacity-25 transition-all duration-700 relative overflow-hidden`}>
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_120%,rgba(255,255,255,0.15),transparent_70%)]" />
        </div>

        <div className="px-6 pb-8 text-center relative flex-grow flex flex-col items-center">
          {/* Avatar Section */}
          <div className="relative -mt-16 mb-6">
            <motion.div
              whileHover={{ scale: 1.1, rotate: 5 }}
              className="relative"
            >
              <div className={`absolute -inset-2 bg-gradient-to-r ${gradient} rounded-full blur-lg opacity-0 group-hover:opacity-50 transition-opacity duration-500`} />
              <img
                src={member.image}
                alt={member.name}
                className={`relative w-32 h-32 rounded-full border-[6px] border-brand-surface shadow-2xl object-cover ${member.imagePosition || "object-center"} ring-1 ring-white/10`}
              />

              {/* Floating Mini Badges */}
              <div className="absolute -bottom-1 left-1/2 -translate-x-1/2 flex gap-1.5">
                {member.category.map((cat) => {
                  const Icon = CatIcon[cat] || Code2;
                  const dotColor = cat === "Frontend" ? "bg-orange-500" : cat === "Backend" ? "bg-blue-500" : "bg-emerald-500";
                  return (
                    <motion.div
                      key={cat}
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      whileHover={{ scale: 1.2, y: -2 }}
                      className={`p-2 ${dotColor} rounded-full text-white shadow-lg border-2 border-brand-surface`}
                    >
                      <Icon size={12} />
                    </motion.div>
                  );
                })}
              </div>
            </motion.div>
          </div>

          {/* Member Info */}
          <motion.h3
            className="text-2xl font-bold text-white mb-1 group-hover:text-transparent group-hover:bg-clip-text group-hover:bg-gradient-to-r group-hover:from-white group-hover:to-brand-accent transition-all duration-300"
          >
            {member.name}
          </motion.h3>
          <p className="text-sm text-brand-accent/80 mb-6 font-semibold uppercase tracking-widest flex items-center gap-2">
            <span className="h-px w-4 bg-brand-accent/30" />
            {member.role}
            <span className="h-px w-4 bg-brand-accent/30" />
          </p>

          {/* Skills Grid */}
          <div className="flex flex-wrap justify-center gap-2 mb-8">
            {member.skills.map((skill, sIdx) => (
              <motion.span
                key={skill}
                initial={{ opacity: 0, y: 10 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 + (sIdx * 0.05) }}
                whileHover={{ scale: 1.05, backgroundColor: "rgba(255,255,255,0.05)" }}
                className={`px-3 py-1.5 text-[10px] sm:text-[11px] uppercase tracking-tighter rounded-xl font-black border transition-color duration-300 shadow-sm ${badgeColors[primaryCat] || badgeColors.Frontend}`}
              >
                {skill}
              </motion.span>
            ))}
          </div>

          {/* Social Actions */}
          <div className="flex justify-center gap-4 mt-auto pt-4 border-t border-white/5 w-full">
            <SocialLink href={member.socials.github} icon={Github} label="GitHub" color="hover:text-white" />
            <SocialLink href={member.socials.linkedin} icon={Linkedin} label="LinkedIn" color="hover:text-blue-400" />
            <SocialLink href={member.socials.portfolio} icon={Globe} label="Website" color="hover:text-brand-accent" />
          </div>
        </div>
      </div>
    </motion.div>
  );
};

const SocialLink = ({ href, icon: Icon, label, color }) => (
  <motion.a
    href={href}
    target="_blank"
    rel="noopener noreferrer"
    whileHover={{ scale: 1.2, y: -3 }}
    whileTap={{ scale: 0.9 }}
    className={`p-2.5 rounded-2xl bg-white/5 text-brand-text-muted ${color} hover:bg-white/10 border border-white/5 transition-all duration-300`}
    title={label}
  >
    <Icon size={18} />
  </motion.a>
);

/* -- Data -- */
const teamMembers = [
  {
    id: 1,
    name: "Krish Dargar",
    role: "Full Stack Developer",
    category: ["Frontend", "Backend"],
    image: "/Profile_Photos/Krish%20Profile.jpeg",
    skills: ["React", "Node.js", "MongoDB", "Tailwind"],
    socials: {
      github: "https://github.com/KD2303",
      linkedin: "https://www.linkedin.com/in/krish-dargar-101774324/",
      portfolio: "#",
    },
  },
  {
    id: 2,
    name: "Sheetal Gourh",
    role: "Frontend Developer",
    category: ["Frontend"],
    image: "/Profile_Photos/Sheetal%20Profile.jpeg",
    skills: ["HTML5", "CSS3", "React", "Figma"],
    socials: {
      github: "https://github.com/sheetal783",
      linkedin: "https://www.linkedin.com/in/sheetal-gourh-994272339/",
      portfolio: "#",
    },
  },
  {
    id: 3,
    name: "Arin Gupta",
    role: "Backend Developer",
    category: ["Backend"],
    image: "/Profile_Photos/Arin%20Profile%20photo.jpeg",
    skills: ["Node.js", "Express", "MongoDB", "REST APIs"],
    socials: {
      github: "https://github.com/arin-gupta06",
      linkedin: "https://www.linkedin.com/in/arin-gupta-2b94b032a/",
      portfolio: "#",
    },
  },
  {
    id: 4,
    name: "Vivek Chaurasiya",
    role: "Backend Developer",
    category: ["Backend"],
    image: "/Profile_Photos/Vivek%20Profile%20Photo.png",
    imagePosition: "object-top",
    skills: ["Node.js", "Express.js", "MongoDB", "REST APIs"],
    socials: {
      github: "https://github.com/VivekChaurasiya95",
      linkedin: "https://www.linkedin.com/in/vivek-chaurasiya-722037315/",
      portfolio: "#",
    },
  },
];

const mentor = {
  name: "Manan Chawla",
  role: "Faculty Mentor / Project Guide",
  image: "/Profile_Photos/Manan%20Sir%20Profile.jpeg",
  description:
    "Supporting the development team with strategic guidance and technical expertise. Fostering a culture of innovation and collaborative excellence.",
  socials: {
    github: "https://github.com/mananchawla26",
    linkedin: "https://www.linkedin.com/in/mananchawla26",

  },
};

const filters = ["All", "Frontend", "Backend", "Mentor"];

const filterIcons = {
  All: Sparkles,
  Frontend: Layers,
  Backend: Database,
  Mentor: Code2,
};

const categoryColor = {
  Frontend: "from-blue-500 to-cyan-400",
  Backend: "from-blue-500 to-cyan-400",
  Testing: "from-blue-500 to-cyan-400",
};

const categoryBadgeBg = {
  Frontend: "bg-blue-500/15 text-blue-400",
  Backend: "bg-blue-500/15 text-blue-400",
  Testing: "bg-blue-500/15 text-blue-400",
};

/* -- Component -- */
const Developers = () => {
  const [activeFilter, setActiveFilter] = useState("All");

  const filteredMembers =
    activeFilter === "All" || activeFilter === "Mentor"
      ? teamMembers
      : teamMembers.filter((m) => m.category.includes(activeFilter));

  const showMentor = activeFilter === "All" || activeFilter === "Mentor";
  const showMembers = activeFilter !== "Mentor";

  return (
    <div className="min-h-screen bg-brand-dark">
      {/* -------- HERO -------- */}
      <section className="relative pt-16 pb-24 overflow-hidden">
        {/* glow blobs */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[600px] bg-brand-accent/10 rounded-full blur-[140px] pointer-events-none" />
        <div className="absolute bottom-0 right-0 w-96 h-96 bg-blue-500/10 rounded-full blur-[120px] pointer-events-none" />

        <div className="wrapper relative z-10 text-center">
          <RevealSection>
            <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-white/5 border border-white/10 text-brand-accent text-sm font-medium mb-6">
              <Code2 size={14} />
              <span>Our Team</span>
            </div>
          </RevealSection>

          <RevealSection delay={100}>
            <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold font-display text-white mb-5 leading-tight">
              Meet Our <span className="text-gradient">Development Team</span>
            </h1>
          </RevealSection>

          <RevealSection delay={200}>
            <p className="text-lg md:text-xl text-brand-text-secondary max-w-2xl mx-auto leading-relaxed">
              The people behind building and improving this project.
            </p>
          </RevealSection>
        </div>
      </section>

      {/* -------- FILTER TABS -------- */}
      <div className="wrapper">
        <RevealSection>
          <div className="flex flex-wrap justify-center gap-3 mb-14">
            {filters.map((f) => {
              const Icon = filterIcons[f];
              const active = activeFilter === f;
              return (
                <button
                  key={f}
                  onClick={() => setActiveFilter(f)}
                  className={`group inline-flex items-center gap-2 px-5 py-2.5 rounded-full text-sm font-medium transition-all duration-300
                    ${active
                      ? "bg-brand-accent text-white shadow-glow scale-105"
                      : "bg-brand-surface border border-brand-border text-brand-text-secondary hover:text-white hover:border-brand-accent/40 hover:scale-105"
                    }`}
                >
                  <Icon
                    size={15}
                    className={
                      active
                        ? "text-white"
                        : "text-brand-text-muted group-hover:text-brand-accent transition-colors"
                    }
                  />
                  {f}
                </button>
              );
            })}
          </div>
        </RevealSection>

        {/* -------- TEAM GRID -------- */}
        {showMembers && (
          <motion.div
            layout
            className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8 mb-24"
          >
            <AnimatePresence mode="popLayout">
              {filteredMembers.map((member, idx) => {
                const primaryCat = member.category[0];
                const gradient = categoryColor[primaryCat] || categoryColor.Frontend;
                return (
                  <DeveloperCard
                    key={member.id}
                    member={member}
                    idx={idx}
                    primaryCat={primaryCat}
                    gradient={gradient}
                  />
                );
              })}
            </AnimatePresence>
          </motion.div>
        )}

        {/* empty filter state */}
        {showMembers && filteredMembers.length === 0 && (
          <div className="text-center py-16">
            <p className="text-brand-text-secondary text-lg">
              No team members found in this category.
            </p>
          </div>
        )}

        {/* -------- FACULTY MENTOR -------- */}
        {showMentor && (
          <RevealSection className="mb-24">
            <motion.div
              whileHover={{ scale: 1.01 }}
              className="relative rounded-[2.5rem] p-px bg-gradient-to-br from-brand-accent/40 via-white/10 to-blue-500/40 overflow-hidden group shadow-2xl"
            >
              {/* animated glow blobs */}
              <div className="absolute top-0 right-0 -mr-24 -mt-24 w-96 h-96 bg-brand-accent/20 rounded-full blur-[100px] opacity-40 animate-float pointer-events-none" />
              <div
                className="absolute bottom-0 left-0 -ml-24 -mb-24 w-96 h-96 bg-blue-500/20 rounded-full blur-[100px] opacity-30 animate-float pointer-events-none"
                style={{ animationDelay: "3s" }}
              />

              <div className="relative bg-brand-surface/90 backdrop-blur-2xl rounded-[39px] p-8 md:p-16 z-10">
                <div className="flex flex-col md:flex-row items-center gap-16">
                  {/* mentor photo */}
                  <div className="relative flex-shrink-0">
                    <motion.div
                      animate={{ y: [0, -10, 0] }}
                      transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
                      className="relative"
                    >
                      <div className="absolute -inset-4 bg-gradient-to-r from-brand-accent to-blue-500 rounded-full blur-2xl opacity-20 group-hover:opacity-40 transition-opacity duration-700" />
                      <div className="absolute -inset-1.5 bg-gradient-to-r from-brand-accent to-blue-500 rounded-full opacity-60 group-hover:opacity-100 transition-opacity duration-700" />
                      <img
                        src={mentor.image}
                        alt={mentor.name}
                        className="relative w-56 h-56 rounded-full border-[8px] border-brand-surface object-cover shadow-2xl ring-1 ring-white/10"
                      />
                    </motion.div>
                  </div>

                  {/* mentor info */}
                  <div className="text-center md:text-left flex-1">
                    <motion.div
                      initial={{ opacity: 0, x: -20 }}
                      whileInView={{ opacity: 1, x: 0 }}
                      className="flex flex-col items-center md:items-start"
                    >
                      <span className="inline-flex items-center gap-2 uppercase tracking-[0.2em] text-[10px] text-brand-accent font-black mb-4 px-4 py-1.5 rounded-full bg-brand-accent/10 border border-brand-accent/20">
                        <Sparkles size={12} />
                        Faculty Mentor
                      </span>
                      <h2 className="text-4xl md:text-5xl font-black font-display text-white mb-3">
                        {mentor.name}
                      </h2>
                      <p className="text-blue-400 text-xl md:text-2xl mb-6 font-bold tracking-tight">
                        {mentor.role}
                      </p>
                      <p className="text-brand-text-secondary text-lg leading-relaxed max-w-2xl mb-10 font-medium">
                        {mentor.description}
                      </p>

                      <div className="flex flex-wrap justify-center md:justify-start gap-5">
                        <motion.a
                          href={mentor.socials.email}
                          whileHover={{ scale: 1.05, boxShadow: "0 0 20px rgba(182,160,255,0.4)" }}
                          whileTap={{ scale: 0.95 }}
                          className="px-8 py-4 bg-brand-accent text-[#020204] font-black rounded-2xl flex items-center gap-3 transition-all duration-300 shadow-lg shadow-brand-accent/20"
                        >
                          <Mail size={18} />
                          Contact Mentor
                        </motion.a>
                        <motion.a
                          href={mentor.socials.linkedin}
                          target="_blank"
                          rel="noopener noreferrer"
                          whileHover={{ scale: 1.1, backgroundColor: "rgba(59, 130, 246, 0.1)", borderColor: "rgba(59, 130, 246, 0.4)", color: "#60a5fa" }}
                          whileTap={{ scale: 0.9 }}
                          className="p-4 rounded-2xl bg-white/5 border border-white/10 text-brand-text-muted transition-all duration-300"
                        >
                          <Linkedin size={24} />
                        </motion.a>
                      </div>
                    </motion.div>
                  </div>
                </div>
              </div>
            </motion.div>
          </RevealSection>
        )}
      </div>
    </div>
  );
};

export default Developers;
