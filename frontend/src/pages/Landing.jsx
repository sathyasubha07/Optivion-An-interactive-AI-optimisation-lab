import { motion } from 'framer-motion'
import HeroVisual from '../components/HeroVisual'
import ModuleGrid from '../components/ModuleGrid'
import { Button, FadeIn, Glass, SectionTitle } from '../components/ui'

const steps = [
  {
    n: '01',
    title: 'Pick a real problem',
    body: 'Each module is a real-world task (an image to sharpen, traffic to classify) posed as an optimisation problem.',
  },
  {
    n: '02',
    title: 'Run the actual solver',
    body: 'Upload your own file or try a sample. The backend runs the module’s own solver code, unmodified.',
  },
  {
    n: '03',
    title: 'Inspect every step',
    body: 'Replay the recorded iterations, read the KKT optimality residuals, and download everything.',
  },
]

const principles = [
  ['Only real numbers', 'Every metric is returned by the solver for your run. Missing values say “not computed”.'],
  ['Replays, labelled', 'Iteration views replay what the solver recorded. Where a solver records nothing, we say so.'],
  ['Maths first', 'Objective, constraints and update rules are shown before you press run, not hidden behind it.'],
]

export default function Landing() {
  return (
    <>
      <section className="mx-auto grid max-w-6xl items-center gap-12 px-4 pb-20 pt-16 sm:px-6 lg:grid-cols-[1.1fr_1fr] lg:pt-24">
        <div>
          <motion.p
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 font-mono text-xs text-mist-300"
          >
            <span className="h-1.5 w-1.5 rounded-full bg-teal" /> An interactive AI optimisation lab
          </motion.p>
          <motion.h1
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.05, ease: [0.22, 1, 0.36, 1] }}
            className="mt-6 font-display text-5xl font-semibold leading-[1.02] tracking-tight sm:text-6xl lg:text-7xl"
          >
            See inside the <span className="text-gradient">black box.</span>
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.15 }}
            className="mt-6 max-w-xl text-lg leading-relaxed text-mist-300"
          >
            OptiVision runs real optimisation algorithms on your data and shows you the whole path:
            the objective, every recorded iteration, and the certificate that says the answer is optimal.
          </motion.p>
          <motion.div
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.25 }}
            className="mt-9 flex flex-wrap gap-3"
          >
            <Button href="#modules">Explore the modules</Button>
            <Button tone="ghost" href="#how">
              How it works
            </Button>
          </motion.div>
        </div>
        <motion.div initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.9, delay: 0.1 }}>
          <HeroVisual />
        </motion.div>
      </section>

      <section id="how" className="mx-auto max-w-6xl scroll-mt-20 px-4 py-16 sm:px-6">
        <FadeIn>
          <SectionTitle eyebrow="How it works" title="From problem to proof in three steps" />
        </FadeIn>
        <div className="mt-10 grid gap-5 md:grid-cols-3">
          {steps.map((s, i) => (
            <FadeIn key={s.n} delay={i * 0.08}>
              <Glass className="h-full p-6">
                <p className="font-mono text-sm text-teal">{s.n}</p>
                <h3 className="mt-3 font-display text-lg font-semibold">{s.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-mist-300">{s.body}</p>
              </Glass>
            </FadeIn>
          ))}
        </div>
      </section>

      <section id="modules" className="mx-auto max-w-6xl scroll-mt-20 px-4 py-16 sm:px-6">
        <FadeIn>
          <SectionTitle eyebrow="Modules" title="Choose a lab">
            Each module poses a real-world task as an optimisation problem and plugs in through one adapter.
          </SectionTitle>
        </FadeIn>
        <div className="mt-10">
          <ModuleGrid />
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-4 py-16 sm:px-6">
        <FadeIn>
          <Glass className="grid gap-8 p-8 md:grid-cols-3 md:p-10">
            {principles.map(([t, b]) => (
              <div key={t}>
                <h3 className="font-display text-lg font-semibold text-mist-100">{t}</h3>
                <p className="mt-2 text-sm leading-relaxed text-mist-300">{b}</p>
              </div>
            ))}
          </Glass>
        </FadeIn>
      </section>
    </>
  )
}
